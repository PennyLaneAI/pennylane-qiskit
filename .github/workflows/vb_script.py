# This script handles the logic to update the changelog and version.py file
# for the automated version bumps
import argparse
import pennylane as qml
pl_version = '"' + qml.version() + '"\n'


def bump_version(version_line, pre_release):
    """ A helper function which takes the current version string and
    replaces it with the bumped version depending on the pre/post
    release flag.

    Args:
         version_line (string): The string containing the current
            version of the plugin.
         pre_release (bool): A flag which determines if this is a
            pre-release or post-release version bump.

    Returns:
        resultant_line (string): A string of the same form as the version line
        with the version number replaced with the bumped version.
        bumped_version (string): The bumped version string.
    """
    data = version_line.split(" ")
    curr_version = data[-1]

    if pre_release:
        bumped_version = pl_version.replace("-dev", "")  # get current Pennylane version then remove -dev
    else:
        split_version = curr_version.split(".")  # "0.17.0" --> ["0,17,0"]
        split_version[1] = str(int(split_version[1]) + 1)  # take middle value and cast as int and bump it by 1
        split_version[2] = split_version[2].replace('"', '-dev"')  # add -dev, ["0,18,0"] --> ["0,18,0-dev"]
        bumped_version = ".".join(split_version)

    data[-1] = bumped_version
    return " ".join(data), bumped_version


def update_version_file(path, pre_release=True):
    """ Updates the __version__ attribute in a specific version file.

    Args:
        path (str): The path to the version file.
        pre_release (bool): A flag which determines if this is a
            pre-release or post-release version bump.

    Return:
        new_version (str): The bumped version string.
    """
    with open(path, 'r', encoding="utf8") as f:
        lines = f.readlines()

    with open(path, 'w', encoding="utf8") as f:
        for line in lines:
            if "__version__" in line.split(' '):
                new_line, new_version = bump_version(line, pre_release)
                f.write(new_line)
            else:
                f.write(line)
    return new_version


def remove_empty_headers(lines):
    """ Takes a paragraph (list of strings) and removes sections which are empty.
    Where a section begins with a header (### Header_Title).

    Args:
        lines (list of strings): the paragraph containing the sections.

    Returns:
        cleaned_lines (list of strings): The paragraph with empty sections removed.
    """
    cleaned_lines = []
    pntr1 = 0

    while pntr1 < len(lines):
        is_empty = True
        for pntr2 in range(pntr1 + 1, len(lines)):
            line2 = lines[pntr2]

            if (len(line2) >= 4) and (line2[:4] == "### "):
                if (pntr1 == 0) or (not is_empty):
                    cleaned_lines.extend(lines[pntr1:pntr2])  # keep these sections!

                pntr1 = pntr2
                is_empty = True  # reset the empty flag

            elif line2 == '\n':
                pass

            else:
                is_empty = False

        cleaned_lines.extend(lines[pntr1:pntr1+1])
        pntr1 += 1

    return cleaned_lines


def update_changelog(path, new_version, pre_release=True):
    """ Updates the Changelog file depending on the pre_release flag.

    Args:
        path (str): The path to the changelog file.
        new_version (str): The bumped version string.
        pre_release (bool): A flag which determines if this is a
            pre-release or post-release version bump.

    Returns:
        None
    """
    with open(path, 'r', encoding="utf8") as f:
        lines = f.readlines()
        end_of_section_index = 0
        for index, line in enumerate(lines):
            if (len(line) >= 3) and (line[:3] == "---"):
                end_of_section_index = index
                break

    with open(path, 'w', encoding="utf8") as f:
        if not pre_release:  # post_release append template to top of the changelog
            with open("./.github/workflows/changelog_template.txt", 'r', encoding="utf8") as template_f:
                template_lines = template_f.readlines()
                template_lines[0] = template_lines[0].replace('x.x.x-dev', new_version)
                f.writelines(template_lines)
                f.writelines(lines)

        else:  # pre_release update the release header and remove any empty headers
            # update release header
            line = lines[0]
            split_line = line.split(" ")
            split_line[-1] = new_version  # replace version (split_line = [#, Release, 0.17.0-dev])
            new_line = " ".join(split_line) + '\n'
            f.write(new_line)

            # remover empty headers
            cleaned_lines = remove_empty_headers(lines[1:end_of_section_index])
            f.writelines(cleaned_lines)

            # keep the rest of the changelog
            rest_of_changelog_lines = lines[end_of_section_index:]
            f.writelines(rest_of_changelog_lines)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version_path", type=str, required=True, help="Path to the _version.py file")
    parser.add_argument("--changelog_path", type=str, required=True, help="Path to the changelog")
    parser.add_argument("--pre_release", dest="release_status", action="store_true",
                        help="True if this is a pre-release version bump, False if it is post release")
    parser.add_argument("--post_release", dest="release_status", action="store_false",
                        help="True if this is a pre-release version bump, False if it is post release")

    args = parser.parse_args()
    updated_version = update_version_file(args.version_path, args.release_status)
    update_changelog(args.changelog_path, updated_version, args.release_status)
