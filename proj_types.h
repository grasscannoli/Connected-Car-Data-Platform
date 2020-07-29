//this file defines all types used in the programme.
#ifndef PROJ_TYPES
#define PROJ_TYPES
class ExpressionNode
{
    public:
        ExpressionNode* left_hand_term;
		std::string column_name;//column name of the expression
		double value;//value of scalar field
        std::string exp_operator;
        ExpressionNode* right_hand_term;
        int type_of_expr;//can be either 1,2 or 3, if bool/decimal/integer
        Expression()
        {
            left_hand_term = right_hand_term = NULL;
            exp_operator = NULL;
        }
        Expression(std::string op)
        {
            left_hand_term = right_hand_term = NULL;
			exp_operator = op;
        }
        double evaluate_double_expr()
        {
				
        }
		bool evaluate_bool_expr()
		{
			if(type_of_expr != 1)
				return false;
			if(right_hand_term != NULL)
			{
				bool x = left_hand_term->evaluate_bool_expr;
				bool y = right_hand_term->evaluate_bool_expr;
				if(exp_operator == "Or")
					return x|y;
				else if(exp_operator == "And")
					return x&y;
			}
			else //left and right are both null. Read value from memory.
			{
					
			}
		}
};
class SelectQuery
{
	public:
		bool distinct;
		vector<std::string> select_columns;
		vector<std::pair<ExpressionNode*,bool>> order_term;
		bool group_term;
		ExpressionNode* group_term;
		bool limit_term;
		int limit_val;
		SQLQuery()
		{
			distinct = false;
			order_term = false;
			group_term = false;
			limit_term = false;
		}
};
#endif