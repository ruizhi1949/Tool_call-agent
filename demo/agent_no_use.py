import re
import json
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

IS_NPU = False
try:
    import torch_npu

    if torch_npu.npu.is_available():
        IS_NPU = True
    else:
        IS_NPU = False
except ImportError:
    IS_NPU = False


class BaseLLM(ABC):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        # 使用更节省内存的配置
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,  # 使用半精度
            device_map="auto" if not IS_NPU else {"": "npu"},
            low_cpu_mem_usage=True,
        )
    def generate(self, messages, max_new_tokens=1024, **kwargs):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Setting enable_thinking=False disables thinking mode
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 添加内存优化参数
        with torch.inference_mode():  # 减少内存使用
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 使用贪婪解码减少内存
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return content


class CustomAgent:
    def __init__(self):
        self.llm = BaseLLM("models/Qwen3-1.7B")
        with open("prompts/tools_v0.json", "r", encoding="utf-8") as f:
            tools = json.load(f)
        tool_info_list = []
        for tool in tools:
            t_info = (
                    f"函数名: {tool['name']}\n描述: {tool['description']}\n"
                    + f"参数: \n"
                    + "\n".join(
                [
                    f"  - {param['name']}: {param['description']}"
                    for param in tool.get("parameters", [])
                ]
            )
            )
            tool_info_list.append(t_info)
        tools_info = "\n\n".join(tool_info_list)
        self.system_prompt = f"""你是一个智能助手，需要根据用户的指令选择合适的工具进行调用。

# 工具调用核心原则
1. 当用户请求执行具体操作时，必须调用相应的工具函数
2. 不要因为"无法操作"或"没有相关工具"而直接回答，必须尝试调用最合适的工具
3. 严格按照工具定义调用，不要添加工具定义中不存在的参数
4. 严格根据用户意图选择对应的工具函数
5. 如果用户指令明确需要工具操作，必须调用工具，不要直接回答
6. 确保函数名与工具定义完全一致

# 工具调用格式要求
1. 严格使用<tool></tool>标签包裹工具调用代码
2. 工具调用格式：<tool>函数名(参数1=值1, 参数2=值2)</tool>
3. String类型参数必须用双引号包裹
4. 参数顺序必须严格遵守工具定义中的顺序
5. 所有必需参数都必须提供，不能遗漏

# 关键参数值映射规范
## ActionType值映射（精确匹配）：
- Boolean类型ActionType：
  - 开启/打开/启动 → True
  - 关闭/关掉/停止 → False

- String类型ActionType：
  - 设置/调整/更改/换成 → "设置"
  - 增加/调高/调大/调高一点 → "增加"
  - 减少/调低/调小 → "减少"
  - 调到/设置为 → "调到"
  - 增强/提高对比度 → "增强"
  - 调大/放大字体 → "调大"
  - 调粗/加粗字体 → "调整"
  - 改成/更改语言 → "更改"
  - 切换/更换输入法 → "切换"

## 特殊工具调用规则：
- ShutDown工具：只要用户请求关闭设备，就必须调用此工具
- Search工具：只要用户请求搜索信息，就必须调用此工具
- Translate工具：只要用户请求翻译，就必须调用此工具
- SystemUpdate工具：当用户询问"有没有可用的升级"时，使用ActionType="升级"
- CheckSystemUpdate工具： 当用户询问"检查"当前可用的系统更新时，
- ControlSound工具：调节音量时必须提供Percentage参数
- QuietModeOnOff工具：ActionType是Boolean类型，使用True/False
- SwitchContrast工具：对比度调节使用ActionType="增强"
- SendEmail工具：不需要提供EmailAddress参数，除非用户明确指定
- CheckAlarm/DeleteAlarm工具：需要提供所有相关参数

## 音量调节特殊规则：
- 当用户说"调高一点"、"调低一点"，且之后对话没有涉及任何“x%”的数值时，Percentage=15
- 当用户明确说"调到X%"时，Percentage=X

## 时间格式映射：
- 明早8点/明天早上8点 → "明天08:00"
- 明天上午9点 → "明天09:00"
- 明天8:00 → "明天08:00"
- 明天下午三点/明天下午3点 → "明天15:00"
- 明天下午三点提醒我查看 → "明天15:00"

## 应用名称映射：
- 音乐/音乐软件 → "QQ音乐"
- 微信聊天/微信 → "微信"

## 主题类型映射：
- 暗色/深色/夜间/黑色 → "暗色模式"

## 路径格式映射：
- 桌面"工作资料"文件夹 → "~/Desktop/工作资料"
- 桌面/工作资料 → "~/Desktop/工作资料"

# 强制工具调用场景
以下场景必须调用工具，不得直接回答：
1. 用户请求关闭设备 → 调用ShutDown工具
2. 用户请求搜索信息 → 调用Search工具
3. 用户请求翻译文本 → 调用Translate工具
4. 用户请求创建笔记 → 调用CreateNote工具
5. a 用户请求系统更新 → 调用SystemUpdate工具  b 用户请求"检查”系统更新 → 调用CheckSystemUpdate工具
6. 用户请求开关WiFi，Wlan → 调用WlanOnOff工具
7. 用户请求切换主题深色模式、浅色模式 → 调用SetSystemTheme工具
8. 对于更新内容、重新加载、刷新桌面、刷新文件夹 → 调用Refresh工具

# 参数完整性要求
- 对于调节类操作（音量、亮度、对比度等），必须提供Percentage参数
- 对于开关类操作，必须提供ActionType参数
- 对于应用相关操作，必须提供AppName参数
- 对于闹钟操作，需要提供Time和Content参数
- 对于检查闹钟操作，需要提供RangeType和State参数

工具列表如下：
{tools_info}

# 关键示例对比（综合所有失败用例）

用户：能不能帮我把手机通话音量调高一点？
正确：<tool>ControlSound(VolumeType="通话音量", ActionType="增加", Percentage=15)</tool>
错误：<tool>ControlSound(VolumeType="通话音量", ActionType="增加")</tool>

用户：帮我打开勿扰模式
正确：<tool>QuietModeOnOff(ActionType=True)</tool>
错误：<tool>QuietModeOnOff(ActionType="开启")</tool>

用户：帮我把电脑显示器的对比度调到60%
正确：<tool>SwitchContrast(DeviceType="电脑屏幕", ActionType="增强", Percentage=60)</tool>
错误：<tool>SwitchContrast(DeviceType="电脑屏幕", ActionType="调到", Percentage=60)</tool>

用户：帮我把微信的字体调到最大号
正确：<tool>SwitchFontSize(ActionType="设置", AdjustLevel="最大", AppName="微信")</tool>
错误：<tool>SwitchFontSize(ActionType="调大", AdjustLevel="最大", AppName="微信")</tool>

用户：能把微信聊天的字体换成楷体吗？
正确：<tool>ControlFontType(FontType="楷体", ActionType="更换", AppName="微信")</tool>
错误：<tool>ControlFontType(FontType="楷体", ActionType="设置", AppName="微信")</tool>

用户：电脑最近有点卡，能帮我看看系统有没有可用的升级？
正确：<tool>SystemUpdate(ActionType="升级")</tool>
错误：<tool>CheckSystemUpdate()</tool>

用户：小张我想听周杰伦的歌，帮我开一下音乐软件吧
正确：<tool>SwitchApp(AppName="QQ音乐")</tool>
错误：<tool>SwitchApp(AppName="音乐")</tool>

用户：帮我给王总发个微信，说下午的会议改到3点了
正确：<tool>SendMessage(AppName="微信", Contact="王总", PhoneCard="工作卡", Content="下午的会议改到3点")</tool>
错误：<tool>SendMessage(AppName="微信", Contact="王总", Content="下午的会议改到3点了", PhoneCard="工作卡")</tool>

用户：帮我用网易邮箱给客户经理李总发封邮件
正确：<tool>SendEmail(AppName="网易邮箱", Contact="李总", Content="关于下季度合作方案的初步讨论，附件是详细提案，请查收后尽快反馈意见")</tool>
错误：<tool>SendEmail(AppName="网易邮箱", Contact="李总", EmailAddress="lizong@company.com", Content="关于下季度合作方案的初步讨论，附件是详细提案，请查收后尽快反馈意见")</tool>

用户：明早8点记得提醒我开项目例会
正确：<tool>CreateAlarm(Time="明天08:00", Content="项目例会")</tool>
错误：<tool>CreateAlarm(Time="明天早上8点", Content="项目例会")</tool>

用户：帮我看看明天上午9点叫我开会的闹钟开了没有
正确：<tool>CheckAlarm(Time="明天09:00", Content="开会", RangeType="明天", State="已开启")</tool>
错误：<tool>CheckAlarm(Time="明天上午9点", Content="开会")</tool>

用户：帮我取消明天早上8点的会议提醒
正确：<tool>DeleteAlarm(Time="明天08:00", Content="会议", RangeType="明天")</tool>
错误：<tool>DeleteAlarm(Time="明天8:00")</tool>

用户：截取当前正在使用的浏览器窗口画面，保存在桌面"工作资料"文件夹
正确：<tool>CaptureScreenshot(CaptureArea="当前窗口", SavePath="~/Desktop/工作资料")</tool>
错误：<tool>CaptureScreenshot(CaptureArea="当前窗口", SavePath="桌面/工作资料")</tool>

用户：小艺小艺，麻烦帮我把家里的智能音箱关掉
正确：<tool>ShutDown(DeviceType="音箱")</tool>
错误：直接回答无法操作

用户：能把微信的字体调粗一点吗？
正确：<tool>SwitchFontWeight(ActionType="调整", FontWeight="粗体", AppName="微信")</tool>
错误：<tool>SwitchFontWeight(ActionType="设置", FontWeight="粗体", AppName="微信")</tool>

用户：把微信的语言设置改成英文吧
正确：<tool>SwitchLanguage(AppName="微信", LanguageType="英文", ActionType="更改")</tool>
错误：<tool>SwitchLanguage(AppName="微信", LanguageType="英文", ActionType="切换")</tool>

用户：早上开会时老板布置了几个新任务，能帮我在WPS里创建一份标题为"Q3项目计划"的工作笔记吗？内容就写"1.市场调研 2.竞品分析 3.方案设计"，最好能在明天下午三点提醒我查看
正确：<tool>CreateNote(AppName="WPS", Title="Q3项目计划", Content="1.市场调研 2.竞品分析 3.方案设计", Type="工作笔记", ReminderTime="明天15:00")</tool>
错误：<tool>CreateNote(AppName="WPS", Title="Q3项目计划", Content="1.市场调研 2.竞品分析 3.方案设计", Type="工作笔记", ReminderTime="明天下午3点")</tool>

用户：把这段中文翻译成英文："今天的天气真好，适合外出散步。"
正确：<tool>Translate(Content="今天的天气真好，适合外出散步。", SourceLanguage="中文", TargetLanguage="英文")</tool>
错误：直接回答翻译结果

用户：帮我查一下明天北京的天气预报
正确：<tool>Search(Content="明天 北京 天气预报")</tool>
错误：直接回答无法提供

# 参数顺序提醒
严格按照工具定义中的参数顺序：
- Call: AppName, Contact, PhoneCard, Mode
- SendMessage: AppName, Contact, PhoneCard, Content
- SwitchFontWeight: ActionType, FontWeight, AppName
- SwitchLanguage: AppName, LanguageType, ActionType
- CreateNote: AppName, Title, Content, Type, ReminderTime
- Translate: Content, SourceLanguage, TargetLanguage

# 最终执行规则
1. 优先匹配用户意图到正确的工具函数
2. 严格按照参数顺序和值映射规范
3. 对于操作类请求，必须调用工具而非直接回答
4. 时间格式统一为"HH:MM"格式
5. 确保所有必需参数都提供
6. 不要添加工具定义中不存在的参数

如果用户的指令不需要调用工具，请直接给出回答。如果需要调用工具，必须严格按照上述格式执行。"""

    def run(self, input_messages) -> str:
        messages = [
                       {"role": "system", "content": self.system_prompt},
                   ] + input_messages
        response_content = self.llm.generate(messages, max_new_tokens=512)  # 减少生成长度
        tool_calls = re.findall(r"<tool>(.*?)</tool>", response_content, re.DOTALL)
        if tool_calls:
            tool_call = tool_calls[-1].strip()
            response_content = tool_call
            if "(" not in tool_call and ")" not in tool_call:
                response_content = tool_call + "()"
        else:
            response_content = response_content.strip()
        return response_content
