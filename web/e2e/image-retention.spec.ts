import { expect, test } from "@playwright/test";
import { defaultSettings } from "../src/api";

test("inference without save controls preserves temporary input preview", async ({
  page,
}) => {
  const id = "b".repeat(32),
    uploadId = "a".repeat(32);
  let submitted: Record<string, unknown> | undefined;
  const job = {
    id,
    kind: "predict",
    title: "仅推理",
    status: "succeeded",
    save_images: false,
    created_at: new Date().toISOString(),
    input_count: 1,
    inputs: [
      {
        id: uploadId,
        name: "sample.png",
        path: `C:\\runs\\web\\jobs\\${id}\\inputs\\${uploadId}.png`,
        url: null,
      },
    ],
    result: {
      report_status: "skipped",
      predictions: [
        {
          input_id: uploadId,
          label: "rose",
          probabilities: { rose: 0.9, daisy: 0.1 },
        },
      ],
    },
  };
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/events") return route.abort();
    if (url.pathname === "/api/predict") {
      submitted = route.request().postDataJSON();
      return route.fulfill({ status: 202, json: job });
    }
    const responses: Record<string, unknown> = {
      "/api/catalog": { models: [], datasets: [] },
      "/api/jobs": submitted ? [job] : [],
      [`/api/jobs/${id}`]: job,
      "/api/runs": [
        {
          id: "test-model",
          name: "test",
          metrics: [],
          classes: [],
          reports: [],
          images: [],
          status: {},
        },
      ],
      "/api/uploads": { id: uploadId, name: "sample.png", url: "/sample.png" },
      "/api/status": {
        settings: defaultSettings,
        scheduler: {
          limits: defaultSettings.concurrency,
          running: { train: 0, predict: 0, auxiliary: 0 },
          queued: { train: 0, predict: 0, auxiliary: 0 },
        },
        system: { cpu_percent: 1, memory_percent: 20 },
        server_time: new Date().toISOString(),
      },
    };
    return route.fulfill({ json: responses[url.pathname] ?? {} });
  });
  await page.addInitScript(() =>
    localStorage.setItem("cls.predict.saveImages", "true"),
  );
  await page.goto("/#/predict");
  await expect(page.getByLabel("保存推理图片及报告")).toHaveCount(0);
  await page.getByLabel("选择推理图片").setInputFiles({
    name: "sample.png",
    mimeType: "image/png",
    buffer: Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aG1kAAAAASUVORK5CYII=",
      "base64",
    ),
  });
  await expect(page.locator(".upload-grid img")).toHaveCount(1);
  await page.getByRole("button", { name: "开始推理", exact: true }).click();
  await expect(page.locator(".prediction-winner strong")).toHaveText("rose");
  expect(submitted?.save_images).toBe(false);
  const preview = page.getByRole("img", {
    name: "输入图片：sample.png",
    exact: true,
  });
  await expect(preview).toBeVisible();
  await expect(preview).toHaveAttribute("src", /^blob:/);
  expect(
    await preview.evaluate((image: HTMLImageElement) => image.naturalWidth),
  ).toBeGreaterThan(0);
  await expect(page.locator(".source-path")).toHaveText(job.inputs[0].path);
  await expect(
    page.getByText("临时推理路径（文件已清理）", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "图片结果：表格视图", exact: true })
    .click();
  await expect(preview).toBeVisible();
  await expect(page.locator(".source-path")).toHaveText(job.inputs[0].path);
  await expect(
    page.getByText("仅推理：未保存图片及报告", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "图片推理", exact: true }).click();
  await expect(page.locator(".upload-grid img")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "开始推理", exact: true }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "任务结果", exact: true }).click();
  await expect(preview).toBeVisible();
  await page.reload();
  await expect(page.getByText("未保存原图", { exact: true })).toBeVisible();
  await expect(page.locator(".source-path")).toHaveText(job.inputs[0].path);
});
