import os




def get_db_profile():
    # Create a file to store the db profile if it doesn't exist
    with open("db_profile.txt", "a", encoding="utf-8") as _:
        pass

    dbProfile = None
    db_profiles = [line for line in open("db_profile.txt", "r", encoding="utf-8") if line.strip()]
    if not(db_profiles):
        dbProfile = input("Enter a profile name for the database (e.g. Tom): ").strip()
    else:
        print("Available profiles:")
        for i, profile in enumerate(db_profiles):
            print(f"{i + 1}. {profile.strip()}")

        choice = input("Select a profile by number or enter a new one: ")
        if (choice.isdigit() and (1 <= int(choice) <= len(db_profiles))):
            return db_profiles[int(choice) - 1].strip()
        else:
            dbProfile = choice.strip()

    with open("db_profile.txt", "a", encoding="utf-8") as f:
        f.write(f"{dbProfile}\n")

    return dbProfile


def get_pixels_path(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.json')]
    if not files:
        raise FileNotFoundError(f"No JSON files found in {folder}")
    
    if len(files) == 1:
        return os.path.join(folder, files[0])
    
    print("Available JSON files:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")

    choice = ""
    while not(choice.isdigit() and (1 <= int(choice) <= len(files))):
        choice = input("Select a file by number: ")

    return os.path.join(folder, files[int(choice) - 1])