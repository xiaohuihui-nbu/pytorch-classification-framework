import { expect, test } from "@playwright/test";
import path from "node:path";
import fs from "node:fs";

test("configuration, history, reports and responsive layout", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.setViewportSize({ width: 1440, height: 1080 });
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "开始一次新的实验" }),
  ).toBeVisible();
  const catalog = await (await page.request.get("/api/catalog")).json();
  await expect(page.locator(".model")).toHaveCount(catalog.models.length);
  if (
    catalog.datasets.some(
      (dataset: { id: string }) => dataset.id === "flower_photos_split",
    )
  ) {
    await expect(page.getByLabel("数据目录")).toHaveValue(
      "flower_photos_split",
    );
  }
  await page.getByRole("button", { name: "高级参数" }).click();
  await expect(page.getByLabel("学习率")).toHaveValue("0.0003");
  await expect(page.getByRole("region", { name: "全局状态栏" })).toBeVisible();
  await expect(page.getByLabel("模型训练并发数")).toHaveCount(0);
  await page.screenshot({
    path: "test-results/workbench-desktop.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "实验结果", exact: true }).click();
  const runs = await (await page.request.get("/api/runs")).json();
  const visualized = runs.find(
    (run: { images: string[] }) => run.images.length,
  );
  if (visualized) {
    await page.getByLabel("实验记录").selectOption(visualized.id);
    await expect(page.locator(".report-images img").first()).toBeVisible();
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "训练工作台", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "选择网络模型" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: "test-results/workbench-mobile.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("global settings, persistent theme, system mode and status bar", async ({
  page,
}) => {
  const original = await (await page.request.get("/api/settings")).json();
  try {
    await page.setViewportSize({ width: 1440, height: 1080 });
    await page.goto("/");
    await page.getByRole("button", { name: "设置", exact: true }).click();
    await expect(page.getByLabel("模型训练并发数")).toHaveValue(
      String(original.concurrency.train),
    );
    await page.getByRole("button", { name: "海洋蓝主题" }).click();
    await page.getByRole("button", { name: "深色", exact: true }).click();
    await page.getByRole("button", { name: "保存全局设置" }).click();
    await expect(page.getByRole("status")).toHaveText(
      "全局设置已保存，立即生效",
    );
    await expect(page.locator("html")).toHaveAttribute("data-theme", "ocean");
    await expect(page.locator("html")).toHaveAttribute("data-mode", "dark");
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({
      path: "test-results/settings-dark.png",
      fullPage: false,
    });
    await page.reload();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "ocean");
    await expect(page.locator("html")).toHaveAttribute("data-mode", "dark");
    const second = await page.context().newPage();
    await second.goto("/");
    await expect(second.locator("html")).toHaveAttribute("data-theme", "ocean");
    await second.close();
    await page.getByRole("button", { name: "打开全局设置" }).click();
    await page.getByRole("button", { name: "跟随系统", exact: true }).click();
    await page.getByRole("button", { name: "保存全局设置" }).click();
    await page.emulateMedia({ colorScheme: "light" });
    await expect(page.locator("html")).toHaveAttribute("data-mode", "light");
    await page.emulateMedia({ colorScheme: "dark" });
    await expect(page.locator("html")).toHaveAttribute("data-mode", "dark");
    await page.getByRole("button", { name: "恢复默认值" }).click();
    await expect(page.getByLabel("模型训练并发数")).toHaveValue("2");
    await expect(page.getByLabel("图片推理并发数")).toHaveValue("1");
    await expect(page.getByLabel("评估与报告并发数")).toHaveValue("1");
    await page.getByRole("button", { name: "保存全局设置" }).click();
    await expect(page.locator("html")).toHaveAttribute("data-mode", "dark");
    await expect(page.locator("html")).toHaveAttribute("data-theme", "cyber");
    await page.screenshot({
      path: "test-results/settings-light.png",
      fullPage: true,
    });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.evaluate(() => window.scrollTo(0, 0));
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBeTruthy();
    await expect(
      page.getByRole("region", { name: "全局状态栏" }),
    ).toBeVisible();
    await page.screenshot({
      path: "test-results/settings-mobile.png",
      fullPage: true,
    });
    await page.route("**/api/status", (route) => route.abort());
    await page.route("**/api/events**", (route) => route.abort());
    await page.reload();
    await expect(
      page.getByRole("region", { name: "全局状态栏" }),
    ).toContainText("服务未连接", { timeout: 10_000 });
  } finally {
    await page.request.post("/api/settings", { data: original });
  }
});

test("upload a real flower and run a saved model", async ({ page }) => {
  const root = path.resolve("..");
  const directory = path.join(root, "data/flower_photos_split/test/daisy");
  const bundles = await (await page.request.get("/api/runs")).json();
  test.skip(!bundles.length, "Requires a locally trained bundle");
  test.skip(
    !fs.existsSync(directory),
    "Requires locally prepared flower data and trained bundles",
  );
  await page.goto("/");
  await page.getByRole("button", { name: "图片推理", exact: true }).click();
  await expect(page.getByLabel("已训练模型")).not.toHaveValue("");
  const image = fs
    .readdirSync(directory)
    .find((name) => /\.(jpg|png|jpeg)$/i.test(name))!;
  await page
    .getByLabel("选择推理图片")
    .setInputFiles(path.join(directory, image));
  await expect(page.locator(".upload-grid img")).toHaveCount(1);
  await page.getByRole("button", { name: "开始推理", exact: true }).click();
  await expect(page.locator(".predictions .prediction")).toHaveCount(1, {
    timeout: 50_000,
  });
  await expect(page).toHaveURL(/#\/results\//);
  await expect(page.locator(".task-table")).toHaveCount(0);
  const originalImage = page.getByRole("img", {
    name: `输入图片：${image}`,
    exact: true,
  });
  await expect(originalImage).toBeVisible();
  await expect(page.locator(".source-name")).toHaveText(image);
  await expect(page.locator(".prediction-winner")).toBeVisible({
    timeout: 50_000,
  });
  await expect(page.locator(".source-path")).toContainText("inputs");
  expect(
    await originalImage.evaluate((img: HTMLImageElement) => img.naturalWidth),
  ).toBeGreaterThan(0);
  const sourceBounds = await page.locator(".prediction-source").boundingBox();
  const outputBounds = await page.locator(".prediction-output").boundingBox();
  expect(sourceBounds!.x + sourceBounds!.width).toBeLessThanOrEqual(
    outputBounds!.x + 1,
  );
  await expect(page.locator(".inference-summary")).toContainText("不进行训练");
  await expect(page.locator(".progress-block")).toHaveCount(0);
  await expect(page.getByRole("img", { name: "验证集准确率曲线" })).toHaveCount(
    0,
  );
  await expect(page.getByRole("link", { name: /打开 HTML 报告/ })).toHaveCount(
    0,
  );
  await page.screenshot({
    path: "test-results/prediction.png",
    fullPage: true,
  });
  const predictedLabel = await page
    .locator(".prediction-winner strong")
    .innerText();
  await page
    .getByRole("button", { name: "图片结果：表格视图", exact: true })
    .click();
  await expect(page.locator(".prediction-table tbody tr")).toHaveCount(1);
  await expect(page.locator(".prediction-table tbody tr")).toContainText(
    predictedLabel,
  );
  await expect(originalImage).toBeVisible();
  await page.getByText("查看概率", { exact: true }).click();
  await expect(page.locator(".prediction-table details[open]")).toBeVisible();
  await page.screenshot({
    path: "test-results/prediction-table.png",
    fullPage: true,
  });
  await page
    .getByRole("button", { name: "图片结果：卡片视图", exact: true })
    .click();
  await expect(
    page.locator(".predictions-cards .prediction-comparison"),
  ).toHaveCount(1);
  await page.reload();
  await expect(
    page.getByRole("button", { name: "图片结果：卡片视图", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.screenshot({
    path: "test-results/prediction-cards.png",
    fullPage: true,
  });
  await page
    .getByRole("button", { name: "图片结果：详细视图", exact: true })
    .click();
  await page.reload();
  await expect(page.locator(".source-name")).toHaveText(image);
  await expect(page.getByText("未保存原图", { exact: true })).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await page
    .getByRole("button", { name: "图片结果：表格视图", exact: true })
    .click();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await page
    .getByRole("button", { name: "图片结果：详细视图", exact: true })
    .click();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  const mobileSource = await page.locator(".prediction-source").boundingBox();
  const mobileOutput = await page.locator(".prediction-output").boundingBox();
  expect(mobileSource!.y + mobileSource!.height).toBeLessThanOrEqual(
    mobileOutput!.y + 1,
  );
  await page.screenshot({
    path: "test-results/prediction-mobile.png",
    fullPage: true,
  });
});
