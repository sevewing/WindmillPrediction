Readme Windmill
# Data
The data is structured into different categories which is: prognosis, observations, settlement and masterdata.
A description of each can be found here.

Settlement readings and masterdata, must be handled with caution. Prognosis and observations, can be shared with others since it contains nothing confidential.

## Prognosis
The prognosis data we have is provided by DMI.
There are around 250 prognosis points in a grid formation over Denmark, from which we gain prognosis data.

Each point has a grid id and coordinates for the position, in UTM_32 format.
Theses informations can be found in prognosis/grid_coordinates.csv.

The prognosis files is structuret in the following way:
- Index: Timestamp in UTC
- Columns: Grid ids and "predicted_ahead", which refers to the amount of hours into the future the value was predicted. For an example a prognosis recieved on 2019-01-01 12:00:00, contains up to 55 timestamps, which then is marked with 1 to 55. 2019-02-01 12:00:00 would be marked with 24, since it is 24 hours into the future.
- Values: Depends on the prognosis type.
    - Temperatur: Values are in kelvin
    - Wind speed: Values are in m/s
    - Wind direction: Values are in deegres clockwise, with 0 being North.

### HIR
The old prognosis model DMI used was called HIR, we have data from this model from 2008-2018.

### ENetNea
ENetNea is the newest prognosis model DMI uses, it is more precise than the prevously model HIR.
We have data from this model since 2018-02-22.

## Observations
Data fetched from DMI [weather archive](https://www.dmi.dk/vejrarkiv/).

## Masterdatawind
This file contains all master data for all windturbines.
The file contains alot of column, and i will only describe the relevant ones.

- GSRN: Unique id for a meteringpoint
- Turbine_name: Name of turbine, can be duplicated.
- Turbine_type:
    - H: Household turbine
    - W: Single turbine
    - P: Turbine park
    - M: Turbine in a park
- Parent_GSRN: If Turbine_type is M, this value referes to the parent meteringpoint, since turbines with type M, does not have settlement readings but is aggregated into the parent.
- In_service: Start of production
- Out_service: End of production
- BBR_municipal: BBR code of the municipal in which the turbine is located.
- Placement:
    - Land: Onshore
    - Hav: Offshore
- UTM_x: UTM_32
- UTM_y: UTM_32
- Capacity_kw: Maximum production capacity in kW.
- Rotor_diameter

## Settlement data
Settlement data from windturbines.
Windturbines located in DK1, which is Jylland and Fyn have a resolution of 15M.
Windturbines located in DK2, which is Sjælland, Bornholm, Lolland and Falster have a resolution of 1H.

The columns contains the following:
- GSRN: Unique id for a meteringpoint, this can be used to fetch masterdata.
- TS_ID: Timeseries id, irrelevant in your case.
- VAERDI: The reading from a meteringpoint in kWh.
- TIME_CET: Timestamp in UTC+1.
- Rotor_diameter: In meters.
- Navhub_height: In meters.
- Actor_short_name: Balance responsible party, the actor which the turbine is connected to.
- Valid_to: The timeframe in which the master data is valid
- Valid_from: The timeframe in which the master data is valid