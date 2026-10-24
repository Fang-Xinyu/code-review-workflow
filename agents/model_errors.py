"""User-facing diagnostics derived only from exception types and HTTP status codes."""
from review_agent.llm import ModelRequestError


def describe_model_failure(error):
    names, statuses, seen = set(), [], set()
    current = error
    for _ in range(8):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        names.add(type(current).__name__)
        status = getattr(current, "status_code", None)
        if isinstance(status, int):
            statuses.append(status)
        current = current.__cause__ or current.__context__
    if names & {"APITimeoutError", "TimeoutError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout"}:
        return "等待模型响应超时。请检查模型服务地址、网络或代理连接，稍后再试；这不是功能描述校验失败。"
    if statuses:
        status = statuses[-1]
        messages = {
            400: "模型服务拒绝了请求参数，请核对模型名称与服务接口是否兼容。",
            401: "模型访问密钥未通过验证，请检查密钥及对应的服务地址。",
            402: "模型服务拒绝了账户请求，请检查服务商的账户状态或额度。",
            403: "模型服务拒绝访问，请检查密钥权限和服务商访问限制。",
            404: "找不到模型或接口，请检查模型名称和模型服务地址。",
            429: "模型服务请求过于频繁，请稍后重试。",
        }
        return messages.get(status, "模型服务暂时不可用，请稍后重试。") + f"（服务状态码：{status}）"
    if names & {"APIConnectionError", "ConnectError", "ConnectionError", "ConnectionRefusedError", "gaierror"}:
        return "无法连接模型服务。请检查模型服务地址、网络、代理或域名解析。"
    if isinstance(error, ModelRequestError):
        return "模型请求失败，未取得有效响应。请检查模型服务地址、模型名称和网络连接；这不是功能描述校验失败。"
    return "模型连接测试未完成，请核对模型服务地址和模型名称。"
