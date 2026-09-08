export const APP_CONFIG = {
  name: "医疗健康科普助手",
  disclaimer:
    "本 AI 仅提供健康科普参考，不构成医疗建议；不能诊断疾病、开处方或提供个体化用药方案。紧急情况请立即拨打 120。",
} as const;

export type AppConfig = typeof APP_CONFIG;
