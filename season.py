def bat_rat_season(month):
    month = month.lower()
    if month in ["december", "january", "february"]:
        return "Winter (Bat Season)"
    elif month in ["march", "april", "may"]:
        return "Spring (Rat Season)"
    elif month in ["june", "july", "august"]:
        return "Summer (Bat Season)"
    elif month in ["september", "october", "november"]:
        return "Autumn (Rat Season)"
    else:
        return "Invalid month!"

# Example usage
user_month = input("Enter a month: ")
print(bat_rat_season(user_month))
