class Schema
{
public:
    int vehicle_id;//the process ID of the car serves as the vehicle ID.
    int database_index;//fixed constant for mapping to the database purposes.
    double oil_life_pct;
    double tire_p_rl;
    double tire_p_rr;
    double tire_p_fl;
    double tire_p_fr;
    double batt_volt;
    double fuel_percentage;
    bool accel;
    bool seatbelt;
    bool hard_brake;
    bool door_lock;
    bool gear_toggle;
    bool clutch;
    bool hard_steer;
    double speed;
    double distance;
    Schema(){}
};