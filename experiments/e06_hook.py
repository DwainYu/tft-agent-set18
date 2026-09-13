class MiniAgent:
    SUPPORTED_EVENTS = {"before_llm", "after_llm", "before_tool", "after_tool"}

    def __init__(self):
        self.hooks = {}   # 事件名 -> 函数列表

    def register_hook(self, event_name, func):
        # 如果 event_name 不在 self.hooks 里，创建空列表，然后把 func 追加进去。
        if event_name not in self.SUPPORTED_EVENTS:
            raise ValueError(f"不支持的事件: {event_name}")
        if event_name not in self.hooks:
            self.hooks[event_name] = []
        self.hooks[event_name].append(func)

    def _trigger(self, event_name, *args, **kwargs):
        # 如果 event_name 在 self.hooks 里，依次调用每个函数
        for func in self.hooks.get(event_name, []):
            try:
                func(*args, **kwargs)
            except Exception as error:
                print(f"钩子 {event_name} 执行失败: {error}")

    def run(self):
        self._trigger("before_llm")
        print("调用 LLM")
        self._trigger("after_llm")

        self._trigger("before_tool")
        print("执行工具")
        self._trigger("after_tool")


import time


class LLMTimer:
    def __init__(self):
        self.start_time = None

    def before_llm(self):
        self.start_time = time.perf_counter()

    def after_llm(self):
        if self.start_time is None:
            print("LLM 调用耗时: 无法计算（未触发 before_llm）")
            return

        cost = time.perf_counter() - self.start_time
        print(f"LLM 调用耗时: {cost:.4f} 秒")
        self.start_time = None

agent = MiniAgent()
timer = LLMTimer()
agent.register_hook("before_llm", timer.before_llm)
agent.register_hook("after_llm", timer.after_llm)
agent.run()