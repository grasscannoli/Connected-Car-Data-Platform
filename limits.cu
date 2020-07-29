class Limits
{
    /*
    Sometimes, certain status messages may contain incorrect update information. Anomaly detetion seeks to check if 
    this is actually the case by seeing if a contiguous number of such messages are actually received. One such
    inconsistent message is an anomaly caused by transmission errors rather than an actual update. 
    */
    public: 
        double interval_between_messages = 0.1;//10 messages per second.
        double speed_violation_time = 1/interval_between_messages;//the number of contiguous status messages that indicate a fault.
        double brake_violation_time = 1/interval_between_messages;//hard brake for 10 continuous messages
        double seatbelt_violation_time = 2/interval_between_messages;
        double pressure_violation_time = 2/interval_between_mesages;
        double oil_violation_time = 4/interval_between_messages;
        double door_violation_time = 2/interval_between_messages;
        double fuel_violation_time = 4/interval_between_mesages;
        double steer_violaton_time = 1/interval_between_messages;
        double max_speed = 100;
        double min_pressure = 30;//pounds per square inch
        double min_voltage = 13.7;//for a typical car, voltage while running is 13.7-14.7 volts
        double engine_temperature_max = 104.44;//typical engine temperatures are in the range 195-220 degrees Fahrenheit.
        double engine_temperature_min = 90.56;
        double min_oil_level = 0.4;//minimum admissible oil percentage
        double min_fuel_percentage = 0.2;//minimum fuel percentage allowable 
        Limits(){}
};