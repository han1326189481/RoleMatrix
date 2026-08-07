# 小R（Little R）生图提示词指南 v2

> 依据专业视觉分析（Character Profile V1 + 细节差距分析）整理。
> 配套 LoRA：models/lora_xiaor_real_v4（触发词 `little_r`）
> 特征词已具象化：必须原封不动使用，否则细节（眼镜型号/发型）会漂移。

## 正向 Prompt（固定特征前缀，必带）

```
little_r, young east asian woman, slender soft face, soft oval face, baby face,
black thick-rimmed rectangular glasses, shaggy layered hair, wispy bangs,
fair skin, petite, calm cool vibe
```

各词对应基底特征（不要替换成泛化词）：
| 特征 | 用词（锁定） | 常见错误写法 |
|---|---|---|
| 眼镜 | black **thick-rimmed rectangular** glasses | glasses（会变细金属圆框） |
| 发型 | **shaggy layered** hair, wispy bangs | long black hair（会变顺直厚重齐刘海） |
| 脸 | **slender** soft face, soft oval face | slightly chubby（会变胖脸） |
| 神态 | **calm cool vibe**, expressionless | smiling（会变甜美 AI 脸） |

## 风格/场景后缀（按场景选用）

- 自拍：`portrait, selfie, looking at camera, indoor, warm lighting, cozy bedroom, phone selfie`
- 图书馆/书店：`reading book, library, bookshelf, natural lighting, quiet atmosphere, candid`
- 街道：`night street, city lights, looking at phone, candid, realistic lifestyle photography`
- 全身：`full body, white hoodie with bold black letters, black pleated skirt, white socks, chunky sneakers`
- 摄影风格：`natural lighting, low contrast, film photography, 35mm photography, casual snapshot, realistic`

## Negative Prompt（推理时必带）

```
old woman, child, loli, muscular, skinny, athletic, heavy makeup, fashion model,
revealing clothes, large breasts, blonde hair, colored hair, short hair, pixie cut,
curly hair, high heels, business suit, strong expression, angry, aggressive,
sharp face, pointed chin, male, anime, cartoon, illustration, drawing, 3d render,
worst quality, low quality, bad anatomy, deformed, blurry, watermark, text
```

## 出图后处理（还原"美颜相机"滤镜感）

AI 原生难出"低对比度灰调"氛围，出图后可 img2img 施加：
- 低饱和度（desaturate 10-20%）
- 冷色调（色温偏蓝）
- 轻微柔焦/磨皮（肤色平滑）
- 对比度降低

## 可选增强（不重训 LoRA）

- **IP-Adapter FaceID/Plus**：以基底图（正脸照/2 或全身照/4）为锚点，出图时参考，
  立即把眼镜、发型、神态拉回 80%（对 LoRA 难以锁死的细节有效）
- 换基模（如 ChilloutMix / MajicMix Realistic）需重新训练 LoRA，谨慎

## Character Bible（人设行为习惯，可注入 LLM/图片一致性）

- 喜欢动漫、轻小说和漫画
- 有点社恐，安静、不爱主动说话
- **常戴黑框矩形眼镜（洗澡/运动才摘）**，很少化妆
- 喜欢宽松卫衣、百褶裙和运动鞋，不追求华丽穿搭
- 经常一个人去咖啡馆、图书馆或便利店
- 出门总背一个黑色单肩包
- 喜欢捧着热饮、看书、发呆或低头玩手机
- 神态自然克制，清冷、若有所思，偶尔微微厌世
