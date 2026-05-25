# TaxiPredict Web

当前 `web` 目录是 TaxiPredict 的唯一前端实现，基于 `Next.js 15 + React 19 + Ant Design + Tailwind + Leaflet`。

旧版 `Vite` 前端 `web/taxiPredict` 已被下线；其中仍有价值的 schema 思路已经提炼到 `src/types/dashboard-schema.ts`，用于后续主前端的数据结构统一。

## 项目结构

```text
web/
├── public/
│   ├── dashboard-assets/     # 预处理图、空间热力图、模型输出图片
│   └── dashboard-data.json   # 主前端直接消费的聚合数据
├── scripts/
│   └── prepare-dashboard-data.mjs
├── src/
│   ├── app/
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   └── page.tsx
│   └── types/
│       └── dashboard-schema.ts
├── next.config.mjs
├── package.json
└── tsconfig.json
```

## 启动方式

先安装依赖，然后启动开发环境：

```bash
pnpm install
pnpm dev
```

`pnpm dev` 和 `pnpm build` 都会先执行数据准备脚本，再启动或构建主前端。

## 数据链路

前端展示数据由 `scripts/prepare-dashboard-data.mjs` 生成。

脚本职责：

- 读取项目上游的月度汇总、聚合 CSV、特征分析报告和模型图片
- 将静态图片复制到 `public/dashboard-assets/`
- 生成主前端使用的 `public/dashboard-data.json`

当前页面主入口在 `src/app/page.tsx`，其直接读取 `public/dashboard-data.json`。

## Schema 演进

`src/types/dashboard-schema.ts` 是主前端后续统一数据结构的目标草案，保留了旧前端中仍然值得继承的几类设计：

- `dimensions`：筛选维度定义，如时间、区域、天气类别
- `breakdowns`：按 borough、weekday、category 等维度切片的数据
- `MultiSeries / HeatmapGrid / GeoPointSeries`：可复用的图表数据结构
- `models.items[]`：统一的模型结果容器，避免为单个模型单独扩字段

当前生产数据仍以 `public/dashboard-data.json` 为准，schema 类型文件用于指导后续重构，而不是要求一次性完成切换。

## 常用命令

```bash
pnpm dev
pnpm build
pnpm start
pnpm lint
pnpm prepare-data
```
