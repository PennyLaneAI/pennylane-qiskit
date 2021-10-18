import argparse
import pennylane as qml


def bump_version(version_line, pre_release):
    data = version_line.split(" ")
    curr_version = data[-1]

    if pre_release:
        # bumped_version = (qml.version()).replace("-dev", "")  # get current Pennylane version then remove -dev
        bumped_version = curr_version.replace("-dev", "")  # remove -dev
    else:
        split_version = curr_version.split(".")  # "0.17.0" --> ["0,17,0"]
        split_version[1] = str(int(split_version[1]) + 1)  # take middle value and cast as int and bump it by 1
        split_version[2] = split_version[2].replace('"', '-dev"')  # add -dev, ["0,18,0"] --> ["0,18,0-dev"]
        bumped_version = ".".join(split_version)

    data[-1] = bumped_version
    return " ".join(data), bumped_version


def update_version_file(path, pre_release=True):
    with open(path, 'r') as f:
        lines = f.readlines()

    with open(path, 'w') as f:
        for line in lines:
            if "__version__" in line.split(' '):
                new_line, new_version = bump_version(line, pre_release)
                f.write(new_line)
            else:
                f.write(line)
    return new_version


def remove_empty_headers(lines):
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
    with open(path, 'r') as f:
        lines = f.readlines()
        end_of_section_index = 0
        for index, line in enumerate(lines):
            if (len(line) >= 3) and (line[:3] == "---"):
                end_of_section_index = index
                break

    with open(path, 'w') as f:
        if not pre_release:  # post_release append template to top of the changelog
            with open("./changelog_template.txt", 'r') as template_f:
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


def main(version_file_path, changelog_file_path, pre_release=True):
    new_version = update_version_file(version_file_path, pre_release)
    update_changelog(changelog_file_path, new_version, pre_release)
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
    main(args.version_path, args.changelog_path, args.release_status)
