# 当前数据目录说明

此项目使用自己的 checkpoint、日志和上传目录。CLI 与 Streamlit 使用 SQLite checkpoint；Agent Server 默认使用进程内 MemorySaver。上传文件按 `thread_id` 保存，但当前没有用户身份与 thread 所有权 ACL。
