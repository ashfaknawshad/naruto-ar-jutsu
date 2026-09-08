/**
 * The 12-seal reference chart, toggleable, shared between the recorder and
 * the live classifier view so contributors and testers alike can compare
 * their hands against the canonical poses without a second window open.
 */
export function createReferenceChart(defaultVisible: boolean): { img: HTMLImageElement; toggleBtn: HTMLButtonElement } {
  const img = document.createElement("img");
  img.src = `${import.meta.env.BASE_URL}image.png`;
  img.alt = "Reference chart of the 12 hand seals";
  img.className = defaultVisible ? "reference-chart visible" : "reference-chart";

  // No positioning class here on purpose — a panel-embedded caller (the
  // recorder) wants this to inherit its own button-row layout, while a
  // standalone caller (the live view) adds "reference-chart-toggle" itself.
  const toggleBtn = document.createElement("button");
  toggleBtn.textContent = defaultVisible ? "Hide seal chart" : "Show seal chart";
  toggleBtn.addEventListener("click", () => {
    const visible = img.classList.toggle("visible");
    toggleBtn.textContent = visible ? "Hide seal chart" : "Show seal chart";
  });

  document.querySelector("#app")!.appendChild(img);

  return { img, toggleBtn };
}
