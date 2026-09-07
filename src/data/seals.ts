/** The 12 zodiac hand seals, plus the two housekeeping classes. */
export const SEALS = [
  "rat",
  "ox",
  "tiger",
  "hare",
  "dragon",
  "snake",
  "horse",
  "ram",
  "monkey",
  "bird",
  "dog",
  "boar",
] as const;

export type Seal = (typeof SEALS)[number];

/** `transition` covers hand movement between seals in a sequence — without
 * it, moving from one seal to the next resets a multi-seal chain. `none` is
 * everything else: idle hands, waving, near-misses. */
export const AUX_LABELS = ["transition", "none"] as const;

export const ALL_LABELS = [...SEALS, ...AUX_LABELS] as const;
export type Label = (typeof ALL_LABELS)[number];
