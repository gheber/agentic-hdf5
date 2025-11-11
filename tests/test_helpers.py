"""
Helper functions for AHDF5 tool unit tests
"""


def validate_success(result):
    """Helper function to check success in result dictionaries"""
    if not 'success' in result:
        print("Result does not contain 'success' key:", result)
        assert False

    if result['success'] != True:
        print("Operation was not successful. Result:", result)

        if 'error' in result:
            print("Error message:", result['error'])

        assert False

    # Success
    return
