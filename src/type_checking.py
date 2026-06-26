from typeguard import TypeCheckError, check_type


def type_matches(obj, type) -> bool:
    try:
        check_type(obj, type)
    except TypeCheckError:
        return False
    else:
        return True