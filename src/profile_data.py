from load_data import load_all

tables = load_all()

traffic = tables["traffic"]
routes_weather = tables["routes_weather"]
schedule = tables["truck_schedule"]
drivers = tables["drivers"]
trucks = tables["trucks"]

print("=" * 60)
print("TRAFFIC")
print("=" * 60)

print("Rows:", len(traffic))
print("Unique routes:", traffic["route_id"].nunique())

traffic_per_route = traffic.groupby("route_id").size()

print("Min rows/route :", traffic_per_route.min())
print("Mean rows/route:", round(traffic_per_route.mean(), 2))
print("Max rows/route :", traffic_per_route.max())

print()

print("=" * 60)
print("ROUTES WEATHER")
print("=" * 60)

print("Rows:", len(routes_weather))
print("Unique routes:", routes_weather["route_id"].nunique())

weather_per_route = routes_weather.groupby("route_id").size()

print("Min rows/route :", weather_per_route.min())
print("Mean rows/route:", round(weather_per_route.mean(), 2))
print("Max rows/route :", weather_per_route.max())

print()

print("=" * 60)
print("TRUCK SCHEDULE")
print("=" * 60)

duration = (
    schedule["estimated_arrival"]
    - schedule["departure_date"]
).dt.total_seconds() / 3600

print("Trips:", len(schedule))
print("Shortest trip (hrs):", duration.min())
print("Average trip (hrs):", round(duration.mean(), 2))
print("Longest trip (hrs):", duration.max())

print()

print("=" * 60)
print("KEY MATCHES")
print("=" * 60)

print("Truck IDs in schedule:", schedule["truck_id"].nunique())
print("Truck IDs in trucks :", trucks["truck_id"].nunique())
print("Vehicle numbers in drivers:", drivers["vehicle_no"].nunique())


schedule_per_route = schedule.groupby("route_id").size()

print("Trips per route")
print("----------------")
print("Min :", schedule_per_route.min())
print("Mean:", round(schedule_per_route.mean(), 2))
print("Max :", schedule_per_route.max())