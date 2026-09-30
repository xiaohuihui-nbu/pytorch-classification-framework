import { expect, test } from "@playwright/test";

test("experiment comparison preserves selection and missing metrics", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const runs = [
    {
      id: "comparison-a",
      name: "A",
      classes: ["red", "blue"],
      status: {},
      reports: [],
      images: [],
      metrics: [
        { epoch: 0, accuracy_top1: 0.7 },
        { epoch: 1, accuracy_top1: 0.8 },
      ],
      config: { model: { name: "tiny_cnn" }, optimizer: { lr: 0.001 } },
    },
    {
      id: "comparison-b",
      name: "B",
      classes: ["red", "blue"],
      status: {},
      reports: [],
      images: [],
      metrics: [{ epoch: 0, accuracy_top1: 0.6 }, { epoch: 1 }],
      config: { model: { name: "tiny_cnn" }, optimizer: { lr: 0.002 } },
    },
  ];
  await page.route("**/api/runs", (route) => route.fulfill({ json: runs }));
  await page.goto("/#/runs");
  const comparison = page.locator(".experiment-comparison");
  await comparison.getByRole("checkbox", { name: "comparison-a" }).click();
  await expect(
    comparison.getByRole("checkbox", { name: "comparison-a" }),
  ).toBeChecked();
  await comparison.getByRole("checkbox", { name: "comparison-b" }).click();
  await expect(
    comparison.getByRole("checkbox", { name: "comparison-b" }),
  ).toBeChecked();
  await expect(
    page.getByRole("img", { name: "多实验验证准确率对比" }),
  ).toBeVisible();
  await expect(comparison.locator("canvas")).toHaveCount(1);
  await expect(
    comparison.locator("tbody").first().locator("tr").nth(0),
  ).toContainText("80.00%");
  await expect(
    comparison
      .locator("tbody")
      .first()
      .locator("tr")
      .nth(1)
      .locator("td")
      .last(),
  ).toHaveText("—");
  await expect(comparison).toContainText("optimizer.lr");
  await expect(comparison).toContainText("0.002");
  await page.reload();
  await expect(
    comparison.getByRole("checkbox", { name: "comparison-a" }),
  ).toBeChecked();
  await expect(
    comparison.getByRole("checkbox", { name: "comparison-b" }),
  ).toBeChecked();
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: "test-results/comparison-mobile.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});
