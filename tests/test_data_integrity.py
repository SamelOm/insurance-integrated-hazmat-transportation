"""Solver-independent integrity checks for the processed thesis data."""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "master_data.xlsx"


def normalize_ids(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = (
                result[column]
                .astype(str)
                .str.replace(r"\.0$", "", regex=True)
            )
    return result


class TestMasterData(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.facilities = normalize_ids(
            pd.read_excel(WORKBOOK, sheet_name="facilities"),
            ("facility_id",),
        )
        cls.paths = normalize_ids(
            pd.read_excel(WORKBOOK, sheet_name="paths"),
            ("facility_id", "path_id"),
        )
        cls.corridors = normalize_ids(
            pd.read_excel(WORKBOOK, sheet_name="corridors"),
            ("corr_id",),
        )
        cls.segments = pd.read_excel(WORKBOOK, sheet_name="segments")
        cls.corridor_risk = normalize_ids(
            pd.read_excel(WORKBOOK, sheet_name="corridor_path_risk"),
            ("facility_id", "path_id", "corridor_id"),
        )
        cls.region_risk = normalize_ids(
            pd.read_excel(WORKBOOK, sheet_name="region_path_risk"),
            ("facility_id", "path_id", "region_id"),
        )

    def test_expected_workbook_schema(self) -> None:
        sheets = set(pd.ExcelFile(WORKBOOK).sheet_names)
        expected = {
            "facilities",
            "paths",
            "corridors",
            "region risk",
            "corridor_path_risk",
            "segments",
            "region_path_risk",
        }
        self.assertTrue(expected.issubset(sheets))

    def test_case_study_dimensions(self) -> None:
        self.assertEqual(len(self.facilities), 40)
        self.assertEqual(len(self.paths), 400)
        self.assertEqual(len(self.corridors), 8)
        self.assertEqual(len(self.segments), 10)
        self.assertEqual(
            set(self.paths.groupby("facility_id").size().unique()),
            {10},
        )

    def test_demand_and_path_keys(self) -> None:
        self.assertEqual(int(self.facilities["D_f"].sum()), 506_613)
        self.assertTrue((self.facilities["D_f"] > 0).all())
        self.assertFalse(self.facilities["facility_id"].duplicated().any())
        self.assertFalse(
            self.paths.duplicated(subset=["facility_id", "path_id"]).any()
        )
        self.assertEqual(
            set(self.paths["facility_id"]),
            set(self.facilities["facility_id"]),
        )

    def test_core_path_parameters_are_finite_and_positive(self) -> None:
        for column in ("P_fp", "R_fp", "U_fp"):
            values = self.paths[column].to_numpy(dtype=float)
            self.assertTrue(np.isfinite(values).all(), column)
            self.assertTrue((values > 0).all(), column)
        self.assertTrue((self.paths["P_fp"] < 1).all())

    def test_region_risk_decomposes_to_path_risk(self) -> None:
        aggregated = (
            self.region_risk.groupby(["facility_id", "path_id"], as_index=False)[
                "R_fpm"
            ]
            .sum()
            .merge(
                self.paths[["facility_id", "path_id", "R_fp"]],
                on=["facility_id", "path_id"],
                how="outer",
                validate="one_to_one",
                indicator=True,
            )
        )
        self.assertTrue((aggregated["_merge"] == "both").all())
        np.testing.assert_allclose(
            aggregated["R_fpm"],
            aggregated["R_fp"],
            rtol=1e-10,
            atol=1e-12,
        )

    def test_corridor_risk_can_be_normalized_to_path_risk(self) -> None:
        raw = (
            self.corridor_risk.groupby(["facility_id", "path_id"], as_index=False)[
                "R_fpk"
            ]
            .sum()
            .merge(
                self.paths[["facility_id", "path_id", "R_fp"]],
                on=["facility_id", "path_id"],
                how="outer",
                validate="one_to_one",
                indicator=True,
            )
        )
        self.assertTrue((raw["_merge"] == "both").all())
        self.assertTrue((raw["R_fpk"] > 0).all())
        normalized_sum = raw["R_fpk"] * (raw["R_fp"] / raw["R_fpk"])
        np.testing.assert_allclose(
            normalized_sum,
            raw["R_fp"],
            rtol=1e-12,
            atol=1e-14,
        )

    def test_segment_bounds_are_ordered(self) -> None:
        segments = self.segments.sort_values("s")
        self.assertTrue((segments["L"] <= segments["U"]).all())
        self.assertTrue((segments["c"] > 0).all())
        self.assertTrue((segments["s"].to_numpy() == np.arange(1, 11)).all())


if __name__ == "__main__":
    unittest.main()
