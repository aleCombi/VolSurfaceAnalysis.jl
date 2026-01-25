# Local Data Store
# Abstraction over local parquet files for historical options data

using Dates
using DataFrames

"""
    LocalDataStore

Provides access to local parquet files containing historical options data.
Designed to work with data synced from DeribitVols.

# Fields
- `root_path::String`: Root directory containing parquet files
- `file_pattern::Symbol`: How files are organized (:flat, :by_date, :by_underlying)

# Supported Layouts
- `:flat` - All parquet files in root_path (e.g., `data/vols_20260117.parquet`)
- `:by_date` - Files named by date (e.g., `data/2024-12-01.parquet`)
- `:by_underlying` - Partitioned (e.g., `data/BTC/2024-12-01.parquet`)
"""
struct LocalDataStore
    root_path::String
    file_pattern::Symbol
    
    function LocalDataStore(root_path::String; file_pattern::Symbol=:auto)
        isdir(root_path) || error("Directory not found: $root_path")
        
        # Auto-detect pattern if not specified
        if file_pattern == :auto
            file_pattern = detect_file_pattern(root_path)
        end
        
        new(root_path, file_pattern)
    end
end

"""
    detect_file_pattern(root_path) -> Symbol

Auto-detect the file organization pattern in a directory.
"""
function detect_file_pattern(root_path::String)::Symbol
    entries = readdir(root_path)
    
    # Check for underlying subdirectories
    if any(e -> e in ["BTC", "ETH", "btc", "eth"], entries)
        return :by_underlying
    end
    
    # Check for date-named files
    parquet_files = filter(f -> endswith(f, ".parquet"), entries)
    if !isempty(parquet_files)
        # Check if filename contains a date pattern
        sample = first(parquet_files)
        if occursin(r"\d{4}-\d{2}-\d{2}", sample) || occursin(r"\d{8}", sample)
            return :by_date
        end
    end
    
    return :flat
end

"""
    list_parquet_files(store::LocalDataStore) -> Vector{String}

List all parquet files in the data store.
"""
function list_parquet_files(store::LocalDataStore)::Vector{String}
    files = String[]
    
    for (root, _, filenames) in walkdir(store.root_path)
        for f in filenames
            if endswith(f, ".parquet")
                push!(files, joinpath(root, f))
            end
        end
    end
    
    return sort(files)
end

"""
    available_dates(store::LocalDataStore; underlying::Union{Underlying,Nothing}=nothing) -> Vector{Date}

Get all dates for which data is available.
Optionally filter by underlying asset.
"""
function available_dates(store::LocalDataStore; 
                         underlying::Union{Underlying,Nothing}=nothing)::Vector{Date}
    files = list_parquet_files(store)
    dates = Set{Date}()
    
    for file in files
        # Try to extract date from filename
        filename = basename(file)
        
        # Match YYYY-MM-DD or YYYYMMDD patterns
        m = match(r"(\d{4})-(\d{2})-(\d{2})", filename)
        if m !== nothing
            push!(dates, Date(parse(Int, m[1]), parse(Int, m[2]), parse(Int, m[3])))
            continue
        end
        
        m = match(r"(\d{4})(\d{2})(\d{2})", filename)
        if m !== nothing
            push!(dates, Date(parse(Int, m[1]), parse(Int, m[2]), parse(Int, m[3])))
            continue
        end
        
        # If no date in filename, we need to peek at the data
        # Skip for now - caller can use load_all
    end
    
    return sort(collect(dates))
end

"""
    load_file(path::AbstractString) -> Vector{VolRecord}

Load a single parquet file into VolRecords.
"""
function load_file(path::AbstractString)::Vector{VolRecord}
    return read_vol_records(path)
end

"""
    load_all(store::LocalDataStore; underlying::Union{Underlying,Nothing}=nothing) -> Vector{VolRecord}

Load all records from the data store.
Optionally filter by underlying asset.
"""
function load_all(store::LocalDataStore; 
                  underlying::Union{Underlying,Nothing}=nothing)::Vector{VolRecord}
    files = list_parquet_files(store)
    all_records = VolRecord[]
    
    for file in files
        records = load_file(file)
        
        # Filter by underlying if specified
        if underlying !== nothing
            records = filter(r -> r.underlying == underlying, records)
        end
        
        append!(all_records, records)
    end
    
    # Sort by timestamp
    sort!(all_records, by = r -> r.timestamp)
    
    return all_records
end

"""
    load_date(store::LocalDataStore, date::Date; 
              underlying::Union{Underlying,Nothing}=nothing) -> Vector{VolRecord}

Load all records for a specific date.
"""
function load_date(store::LocalDataStore, date::Date;
                   underlying::Union{Underlying,Nothing}=nothing)::Vector{VolRecord}
    # First try to find a file matching this date
    files = list_parquet_files(store)
    date_str1 = Dates.format(date, "yyyy-mm-dd")
    date_str2 = Dates.format(date, "yyyymmdd")
    
    matching_files = filter(files) do f
        name = basename(f)
        occursin(date_str1, name) || occursin(date_str2, name)
    end
    
    records = VolRecord[]
    
    if !isempty(matching_files)
        # Load from date-specific files
        for file in matching_files
            append!(records, load_file(file))
        end
    else
        # Fall back to loading all and filtering
        all_records = load_all(store)
        records = filter(r -> Date(r.timestamp) == date, all_records)
    end
    
    # Filter by underlying if specified
    if underlying !== nothing
        records = filter(r -> r.underlying == underlying, records)
    end
    
    sort!(records, by = r -> r.timestamp)
    return records
end

"""
    load_range(store::LocalDataStore, start_date::Date, end_date::Date;
               underlying::Union{Underlying,Nothing}=nothing) -> Vector{VolRecord}

Load all records within a date range (inclusive).
"""
function load_range(store::LocalDataStore, start_date::Date, end_date::Date;
                    underlying::Union{Underlying,Nothing}=nothing)::Vector{VolRecord}
    records = VolRecord[]
    
    # Try date-by-date loading first
    avail_dates = available_dates(store)
    target_dates = filter(d -> start_date <= d <= end_date, avail_dates)
    
    if !isempty(target_dates)
        for date in target_dates
            append!(records, load_date(store, date; underlying=underlying))
        end
    else
        # Fall back to loading all and filtering
        all_records = load_all(store; underlying=underlying)
        records = filter(r -> start_date <= Date(r.timestamp) <= end_date, all_records)
    end
    
    sort!(records, by = r -> r.timestamp)
    return records
end

"""
    get_timestamps(store::LocalDataStore; 
                   underlying::Union{Underlying,Nothing}=nothing,
                   resolution::Period=Minute(1)) -> Vector{DateTime}

Get all unique timestamps in the data, optionally rounded to resolution.
This is useful for lazy iteration - load only timestamps, then load data on demand.
"""
function get_timestamps(store::LocalDataStore;
                        underlying::Union{Underlying,Nothing}=nothing,
                        resolution::Period=Minute(1))::Vector{DateTime}
    records = load_all(store; underlying=underlying)
    
    timestamps = Set{DateTime}()
    for r in records
        push!(timestamps, floor(r.timestamp, resolution))
    end
    
    return sort(collect(timestamps))
end
