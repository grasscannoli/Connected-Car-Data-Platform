class Limits
{
    /*
    Sometimes, certain status messages may contain incorrect update information. Anomaly detetion seeks to check if 
    this is actually the case by seeing if a contiguous number of such messages are actually received. One such
    inconsistent message is an anomaly caused by transmission errors rather than an acutal update. 
    */
    double violation_time = 10;//the number of contiguous status messages that indicate a fault.
    double max_speed = 80;
    double min_pressure = 28;
    double min_voltage = 13.7;//for a typical car, voltage while running is 13.7-14.7 volts
    double engine_temperature_max = 104.44;//typical engine temperatures are in the range 195-220 degrees Fahrenheit.
    double engine_temperature_min = 90.56; 
};