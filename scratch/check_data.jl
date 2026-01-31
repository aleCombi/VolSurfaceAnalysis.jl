using DuckDB

conn = DBInterface.connect(DuckDB.DB)

parquet_path = "C:/repos/DeribitVols/data/massive_parquet/minute_aggs/date=2026-01-28/underlying=SPY/data.parquet"

open(joinpath(@__DIR__, "data_check_output.txt"), "w") do io
    # Count calls vs puts for 14:00 timestamp
    println(io, "CALLS VS PUTS COUNT (all data for this date):")
    println(io, "-"^60)
    result = DBInterface.execute(conn, """
        SELECT 
            CASE WHEN ticker LIKE '%C0%' THEN 'CALL' ELSE 'PUT' END as option_type,
            COUNT(*) as cnt
        FROM read_parquet('$parquet_path') 
        GROUP BY 1
    """)
    for row in result
        println(io, "  $(row[1]): $(row[2])")
    end

    # Sample data showing both calls and puts
    println(io, "\n\nSAMPLE CALLS (5 rows):")
    println(io, "-"^60)
    result2 = DBInterface.execute(conn, """
        SELECT ticker, close, volume, timestamp
        FROM read_parquet('$parquet_path') 
        WHERE ticker LIKE '%C0%'
        LIMIT 5
    """)
    for row in result2
        println(io, "  $(row[1]) | close=$(row[2]) | vol=$(row[3]) | $(row[4])")
    end

    println(io, "\n\nSAMPLE PUTS (5 rows):")
    println(io, "-"^60)
    result3 = DBInterface.execute(conn, """
        SELECT ticker, close, volume, timestamp
        FROM read_parquet('$parquet_path') 
        WHERE ticker LIKE '%P0%'
        LIMIT 5
    """)
    for row in result3
        println(io, "  $(row[1]) | close=$(row[2]) | vol=$(row[3]) | $(row[4])")
    end
end

DBInterface.close!(conn)
println("Output written to data_check_output.txt")
