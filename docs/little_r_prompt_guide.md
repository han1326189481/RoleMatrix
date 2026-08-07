# 小R（Little R）生图提示词指南

> 依据专业视觉分析（Character Profile V1）整理，供生图推理使用。
> 配套 LoRA：models/lora_xiaor_real_v2（触发词 `little_r`）

## 固定特征前缀（Prompt 必带）

```
little_r, young east asian woman, slightly chubby, soft round face, baby face,
black rectangular glasses, long black layered hair, wispy bangs, fair skin
```

说明：
- **slightly chubby 必须保留**——防止模型往纸片人方向跑
- **black rectangular glasses 是小R 签名特征**（训练数据中眼镜样本已强化）

## Negative Prompt（推理时必带）

```
old woman, child, loli, muscular, skinny, athletic, heavy makeup, fashion model,
revealing clothes, large breasts, blonde hair, colored hair, short hair, pixie cut,
curly hair, high heels, business suit, strong expression, angry, aggressive,
sharp face, pointed chin, male, anime, cartoon, illustration, drawing, 3d render,
worst quality, low quality, bad anatomy, deformed, blurry, watermark, text
```

## 风格后缀（按场景选用）

- 自拍：`portrait, selfie, looking at camera, indoor, warm lighting, cozy bedroom, phone selfie`
- 图书馆/书店：`reading book, library, bookshelf, natural lighting, quiet atmosphere, candid`
- 街道：`night street, city lights, looking at phone, candid, realistic lifestyle photography`
- 全身：`full body, oversized sweatshirt, black pleated skirt, white socks, chunky sneakers, candid`
- 摄影风格：`natural lighting, soft lighting, warm lighting, film photography, 35mm photography, casual snapshot, lifestyle photography, realistic`

## Character Bible（人设行为习惯，可注入 LLM/图片一致性）

- 喜欢动漫、轻小说和漫画
- 有点社恐，安静、不爱主动说话
- 常戴黑框眼镜，很少化妆
- 喜欢宽松卫衣、百褶裙和运动鞋，不追求华丽穿搭
- 经常一个人去咖啡馆、图书馆或便利店
- 出门总背一个黑色单肩包
- 喜欢捧着热饮、看书、发呆或低头玩手机
- 神态自然克制，害羞、认真、若有所思
