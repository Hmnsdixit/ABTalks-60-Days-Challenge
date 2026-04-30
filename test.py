def test_prediction():
    values = [2, 4, 6]
    avg = sum(values) / len(values)

    if avg > 5:
        result = "High"
    elif avg > 2:
        result = "Medium"
    else:
        result = "Low"

    assert result == "Medium"

# Call function
test_prediction()

print("Test Passed ✅ CI/CD Working 🚀")