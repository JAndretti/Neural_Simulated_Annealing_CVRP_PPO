def main():
    val = "1000"

    with open(f"generated_problem/scripts_gen/genXML{val}.sh", "w") as output_file:
        with open("bdd/Vrp-Set-XML100/genXML100.sh", "r") as file:
            for line in file:
                lst = line.strip().split()
                if lst and lst[0] == "#":
                    continue
                elif lst[0] == "python":  # Check if the line is not empty
                    lst[2] = val
                    lst[1] = f"generated_problem/bdd_generated/gen{val}/generator.py"
                line = " ".join(lst)
                output_file.write(line + "\n")


if __name__ == "__main__":
    main()
