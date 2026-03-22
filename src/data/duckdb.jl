# DuckDB Query Helpers
# Generic DuckDB <-> DataFrame utilities (no source-specific logic)

using DataFrames
using DuckDB

# ============================================================================
# Core Query Helpers
# ============================================================================

"""
    _duckdb_query_df(query) -> DataFrame

Execute a DuckDB query against an in-memory database and return a DataFrame.
"""
function _duckdb_query_df(query::AbstractString)::DataFrame
    con = DuckDB.DBInterface.connect(DuckDB.DB, ":memory:")
    try
        return DuckDB.DBInterface.execute(con, query) |> DataFrame
    finally
        DuckDB.DBInterface.close!(con)
    end
end

"""
    _duckdb_parquet_df(path; columns="*", where="") -> DataFrame

Read a parquet file via DuckDB and return a DataFrame.
"""
function _duckdb_parquet_df(
    path::AbstractString;
    columns::AbstractString="*",
    where::AbstractString=""
)::DataFrame
    path_sql = replace(String(path), "\\" => "/")
    query = "SELECT $columns FROM '$path_sql'"
    if !isempty(strip(where))
        query *= " WHERE $where"
    end
    return _duckdb_query_df(query)
end
