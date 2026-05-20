from __future__ import annotations


def resolve_limit(
    row_count_limit: int | None,
    random_row_count_limit: int | None,
    n_rows: int | None,
) -> tuple[int | None, int | None]:
    if sum(v is not None for v in (row_count_limit, random_row_count_limit, n_rows)) > 1:
        raise ValueError(
            "Only one of row_count_limit, random_row_count_limit, and n_rows may be set."
        )

    limit = row_count_limit if row_count_limit is not None else n_rows
    random_limit = random_row_count_limit

    if limit is not None and limit < 1:
        raise ValueError("row_count_limit/n_rows must be positive.")
    if random_limit is not None and random_limit < 1:
        raise ValueError("random_row_count_limit must be positive.")

    return limit, random_limit


def where_sql(
    start_date_limit: str | None,
    end_date_limit: str | None,
) -> tuple[str, dict[str, str | int]]:
    parameters: dict[str, str | int] = {}
    clauses: list[str] = []

    if start_date_limit is not None:
        clauses.append("datetime >= parseDateTimeBestEffort({start_dt:String})")
        parameters["start_dt"] = start_date_limit.replace("T", " ")

    if end_date_limit is not None:
        clauses.append("datetime <= parseDateTimeBestEffort({end_dt:String})")
        parameters["end_dt"] = (
            f"{end_date_limit} 23:59:59"
            if len(end_date_limit) == 10
            else end_date_limit.replace("T", " ")
        )

    if not clauses:
        return "", parameters

    return "WHERE " + " AND ".join(clauses), parameters
