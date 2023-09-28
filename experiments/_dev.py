list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]

# Convert lists to sets
set1 = set(list1)
set2 = set(list2)

# Find common and unique elements
common_elements = set1.intersection(set2)
missed_in_list1 = set2.difference(set1)
missed_in_list2 = set1.difference(set2)

# Print the results
print(f"Common elements: {common_elements}")
print(f"Elements in list2 but not in list1: {missed_in_list1}")
print(f"Elements in list1 but not in list2: {missed_in_list2}")

# Print counts
print(f"Number of common elements: {len(common_elements)}")
print(f"Number of elements missed in list1: {len(missed_in_list1)}")
print(f"Number of elements missed in list2: {len(missed_in_list2)}")
