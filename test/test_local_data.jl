@testset "Local Data Store" begin
    # Use existing sample data file
    data_path = joinpath(@__DIR__, "..", "data")
    sample_file = joinpath(data_path, "vols_20260117.parquet")
    
    if isdir(data_path) && isfile(sample_file)
        @testset "LocalDataStore Construction" begin
            store = LocalDataStore(data_path)
            @test store.root_path == data_path
            @test store.file_pattern in [:flat, :by_date, :by_underlying]
        end
        
        @testset "list_parquet_files" begin
            store = LocalDataStore(data_path)
            files = list_parquet_files(store)
            @test length(files) >= 1
            @test all(f -> endswith(f, ".parquet"), files)
        end
        
        @testset "load_all" begin
            store = LocalDataStore(data_path)
            records = load_all(store)
            @test length(records) > 0
            @test all(r -> r isa VolRecord, records)
            
            # Should be sorted by timestamp
            for i in 2:length(records)
                @test records[i].timestamp >= records[i-1].timestamp
            end
        end
        
        @testset "load_all with underlying filter" begin
            store = LocalDataStore(data_path)
            btc_records = load_all(store; underlying=BTC)
            eth_records = load_all(store; underlying=ETH)
            
            @test all(r -> r.underlying == BTC, btc_records)
            @test all(r -> r.underlying == ETH, eth_records)
        end
        
        @testset "available_dates" begin
            store = LocalDataStore(data_path)
            dates = available_dates(store)
            # Should detect date from filename vols_20260117.parquet
            @test Date(2026, 1, 17) in dates
        end
        
        @testset "load_date" begin
            store = LocalDataStore(data_path)
            records = load_date(store, Date(2026, 1, 17))
            @test length(records) > 0
            @test all(r -> Date(r.timestamp) == Date(2026, 1, 17), records)
        end
        
        @testset "get_timestamps" begin
            store = LocalDataStore(data_path)
            ts = get_timestamps(store; resolution=Hour(1))
            @test length(ts) >= 1
            # Should be sorted
            for i in 2:length(ts)
                @test ts[i] > ts[i-1]
            end
        end
    else
        @info "Test data not found at $data_path, skipping LocalDataStore tests"
    end
end

@testset "Surface Iterator" begin
    data_path = joinpath(@__DIR__, "..", "data")
    sample_file = joinpath(data_path, "vols_20260117.parquet")
    
    if isdir(data_path) && isfile(sample_file)
        store = LocalDataStore(data_path)
        
        @testset "SurfaceIterator Construction" begin
            iter = SurfaceIterator(store, BTC)
            @test length(iter) >= 1
            @test iter.underlying == BTC
        end
        
        @testset "Iteration" begin
            iter = SurfaceIterator(store, BTC; resolution=Hour(1))
            surfaces = collect(iter)
            
            @test length(surfaces) >= 1
            @test all(s -> s isa VolatilitySurface, surfaces)
            @test all(s -> s.underlying == BTC, surfaces)
            
            # Surfaces should be chronologically ordered
            for i in 2:length(surfaces)
                @test surfaces[i].timestamp >= surfaces[i-1].timestamp
            end
        end
        
        @testset "Random Access by Index" begin
            iter = SurfaceIterator(store, BTC)
            
            if length(iter) >= 2
                s1 = surface_at(iter, 1)
                s2 = surface_at(iter, 2)
                
                @test s1.timestamp < s2.timestamp
                @test s1.underlying == BTC
            end
        end
        
        @testset "Random Access by Timestamp" begin
            iter = SurfaceIterator(store, BTC; resolution=Hour(1))
            
            if length(iter) >= 1
                ts = first(timestamps(iter))
                surface = surface_at(iter, ts)
                
                @test surface !== nothing
                @test surface.underlying == BTC
            end
        end
        
        @testset "Date Range Filtering" begin
            iter = SurfaceIterator(store, BTC; 
                                   start_date=Date(2026, 1, 17),
                                   end_date=Date(2026, 1, 17))
            
            for surface in iter
                @test Date(surface.timestamp) == Date(2026, 1, 17)
            end
        end
        
        @testset "first/last_timestamp" begin
            iter = SurfaceIterator(store, BTC)
            
            @test first_timestamp(iter) <= last_timestamp(iter)
            @test first_timestamp(iter) == iter.timestamps[1]
            @test last_timestamp(iter) == iter.timestamps[end]
        end
        
        @testset "filter_timestamps" begin
            iter = SurfaceIterator(store, BTC; resolution=Hour(1))
            
            # Filter to only afternoon hours
            afternoon = filter_timestamps(iter, ts -> Dates.hour(ts) >= 12)
            
            for ts in timestamps(afternoon)
                @test Dates.hour(ts) >= 12
            end
        end
    else
        @info "Test data not found at $data_path, skipping SurfaceIterator tests"
    end
end
