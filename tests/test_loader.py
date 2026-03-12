from pathlib import Path

from image_compare.loader import find_images, parse_sequence_number


class TestParseSequenceNumber:
    def test_standard_filename(self):
        assert parse_sequence_number("0_385457_modelname_0001.webp") == 0

    def test_sequence_number_nonzero(self):
        assert parse_sequence_number("5_12345_model_0002.webp") == 5

    def test_multidigit_sequence(self):
        assert parse_sequence_number("12_99999_model_0003.webp") == 12

    def test_no_sequence_returns_none(self):
        assert parse_sequence_number("no_leading_digit.webp") is None

    def test_empty_string_returns_none(self):
        assert parse_sequence_number("") is None

    def test_underscore_only_returns_none(self):
        assert parse_sequence_number("_something.webp") is None


class TestFindImages:
    def test_groups_by_sequence_number(self, tmp_path: Path):
        model_a = tmp_path / "model_a"
        model_b = tmp_path / "model_b"
        model_a.mkdir()
        model_b.mkdir()

        (model_a / "0_111_model_a_0001.webp").touch()
        (model_b / "0_222_model_b_0001.webp").touch()
        (model_a / "1_333_model_a_0001.webp").touch()

        groups = find_images(tmp_path)

        assert set(groups.keys()) == {0, 1}
        assert len(groups[0]) == 2
        assert len(groups[1]) == 1

    def test_sorts_paths_by_folder_name(self, tmp_path: Path):
        z_folder = tmp_path / "z_model"
        a_folder = tmp_path / "a_model"
        z_folder.mkdir()
        a_folder.mkdir()

        (z_folder / "0_111_z_0001.webp").touch()
        (a_folder / "0_222_a_0001.webp").touch()

        groups = find_images(tmp_path)

        assert groups[0][0].parent.name == "a_model"
        assert groups[0][1].parent.name == "z_model"

    def test_empty_directory_returns_empty_dict(self, tmp_path: Path):
        assert find_images(tmp_path) == {}

    def test_ignores_non_matching_files(self, tmp_path: Path):
        (tmp_path / "readme.txt").touch()
        (tmp_path / "image.png").touch()

        assert find_images(tmp_path) == {}

    def test_custom_glob_pattern(self, tmp_path: Path):
        (tmp_path / "0_111_model_0001.png").touch()

        groups = find_images(tmp_path, glob_pattern="*.png")

        assert 0 in groups
