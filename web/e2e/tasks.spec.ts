import { expect, test } from "@playwright/test";

test("task views, filters, rename, clone and bulk delete with UI fixtures", async ({
  page,
}) => {
  // Browser fixtures exercise interactions; real scheduler behavior is covered in web/qa.
  let jobs = [
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "interrupted",
  ].map((status, i) => ({
    id: `fixture-${i}`,
    title: `示例实验 ${i + 1}`,
    status,
    kind: "train",
    created_at: `2026-09-29T0${i}:00:00Z`,
    log: "示例日志",
    metrics: [],
    parameters: {
      model_id: "flower_resnet18",
      dataset_id: "flower_photos_split",
      epochs: 10,
    },
  }));
  await page.route("**/api/jobs**", async (route) => {
    const pathname = new URL(route.request().url()).pathname;
    const method = route.request().method();
    if (pathname === "/api/jobs") return route.fulfill({ json: jobs });
    if (pathname === "/api/jobs/page") {
      const params = new URL(route.request().url()).searchParams;
      const query = params.get("query") || "";
      const filtered = jobs.filter(
        (j) =>
          (!params.get("kind") || j.kind === params.get("kind")) &&
          (!params.get("status") || j.status === params.get("status")) &&
          `${j.title} ${j.id}`.includes(query),
      );
      const offset = Number(params.get("offset") || 0),
        limit = Number(params.get("limit") || 12);
      return route.fulfill({
        json: {
          items: filtered.slice(offset, offset + limit),
          total: filtered.length,
          offset,
          limit,
        },
      });
    }
    if (pathname === "/api/jobs/batch-delete") {
      const body = route.request().postDataJSON();
      expect(body.stop_running).toBe(true);
      jobs = jobs.filter((j) => !body.job_ids.includes(j.id));
      return route.fulfill({
        json: { deleted: body.job_ids, artifacts_preserved: true },
      });
    }
    const id = pathname.split("/")[3];
    const job = jobs.find((j) => j.id === id);
    if (!job)
      return route.fulfill({ status: 404, json: { detail: "任务不存在" } });
    if (pathname.endsWith("/clone")) {
      const copy = {
        ...job,
        id: "fixture-copy",
        title: job.title + "（副本）",
        status: "queued",
      };
      jobs.push(copy);
      return route.fulfill({ status: 202, json: copy });
    }
    if (method === "PATCH") Object.assign(job, route.request().postDataJSON());
    return route.fulfill({ json: job });
  });
  await page.setViewportSize({ width: 1440, height: 1080 });
  await page.goto("/");
  await page.getByRole("button", { name: "任务管理", exact: true }).click();
  await expect(page.locator(".task-table tbody tr")).toHaveCount(6);
  await expect(page.locator(".log-panel")).toHaveCount(0);
  await page
    .getByRole("button", { name: "查看 示例实验 1", exact: true })
    .click();
  await expect(page).toHaveURL(/#\/results\/fixture-0$/);
  await expect(page.locator(".task-manager")).toHaveCount(0);
  await expect(page.locator(".log-panel")).toBeVisible();
  await page
    .getByRole("button", { name: "任务结果：卡片视图", exact: true })
    .click();
  await expect(page.locator(".result-overview-card")).toHaveCount(6);
  await expect(page.locator(".log-panel")).toHaveCount(0);
  await page.getByLabel("搜索任务结果").fill("示例实验 2");
  await expect(page.locator(".result-overview-card")).toHaveCount(1);
  await page.getByLabel("搜索任务结果").fill("");
  await page.screenshot({
    path: "test-results/result-catalog-cards.png",
    fullPage: true,
  });
  await page
    .getByRole("button", { name: "任务结果：表格视图", exact: true })
    .click();
  await expect(page.locator(".result-table tbody tr")).toHaveCount(6);
  await page.reload();
  await expect(
    page.getByRole("button", { name: "任务结果：表格视图", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".result-table tbody tr")).toHaveCount(6);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await page.setViewportSize({ width: 1440, height: 1080 });
  await page
    .getByRole("button", { name: "打开结果 示例实验 1", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "任务结果：详细视图", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.reload();
  await expect(page.getByLabel("选择任务结果")).toHaveValue("fixture-0");
  await expect(page.locator(".log-panel")).toBeVisible();
  await page.getByRole("button", { name: "返回任务管理", exact: true }).click();
  await page.goBack();
  await expect(page.getByLabel("选择任务结果")).toHaveValue("fixture-0");
  await page.goForward();
  await expect(page.locator(".task-table tbody tr")).toHaveCount(6);
  await page.getByLabel("搜索任务").fill("示例实验 3");
  await expect(page.locator(".task-table tbody tr")).toHaveCount(1);
  await page.getByLabel("搜索任务").fill("");
  await page
    .getByRole("button", { name: "编辑 示例实验 3", exact: true })
    .click();
  await page.getByLabel("任务名称", { exact: true }).fill("已重命名实验");
  await page.getByRole("button", { name: "保存任务", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "查看 已重命名实验", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "复制并运行 已重命名实验", exact: true })
    .click();
  await expect(page.locator(".task-table tbody tr")).toHaveCount(7);
  await page.screenshot({
    path: "test-results/tasks-table.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "卡片视图" }).click();
  await expect(page.locator(".task-card")).toHaveCount(7);
  await page.screenshot({
    path: "test-results/tasks-cards.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "看板视图" }).click();
  await expect(page.locator(".task-lane")).toHaveCount(4);
  await expect(page.locator(".lane-running .task-card")).toHaveCount(1);
  await page.screenshot({
    path: "test-results/tasks-board.png",
    fullPage: true,
  });
  await page.getByLabel("选择任务 示例实验 1", { exact: true }).check();
  await page.getByLabel("选择任务 示例实验 2", { exact: true }).check();
  await page.getByRole("button", { name: "批量删除", exact: true }).click();
  await expect(page.getByRole("dialog")).toContainText("运行中");
  await page.getByRole("button", { name: "确认删除", exact: true }).click();
  await expect(page.locator(".task-card")).toHaveCount(5);
  await page.setViewportSize({ width: 390, height: 844 });
  for (const view of ["表格视图", "卡片视图", "看板视图"]) {
    await page.getByRole("button", { name: view }).click();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBeTruthy();
  }
  await page.screenshot({
    path: "test-results/tasks-mobile.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "新建任务", exact: true }).click();
  await page.getByRole("button", { name: /新建训练/ }).click();
  await expect(
    page.getByRole("heading", { name: "开始一次新的实验" }),
  ).toBeVisible();
});
