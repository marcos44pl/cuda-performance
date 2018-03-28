#include "Logger.h"



AbstractLogger::AbstractLogger(std::string name, std::ios_base::openmode  mode)
{
    logFile.open(name.c_str(), mode);
    logFile << "Logs:\n";
}

AbstractLogger::~AbstractLogger()
{
    logFile << "                                      "
               "KONIEC TESTU \n\n";
    logFile.close();
}


ErrorLogger & ErrorLogger::getInstance()
{
    static ErrorLogger instance;
    return instance;
}

ErrorLogger::ErrorLogger() : AbstractLogger("errors",std::ios::out)
{
}

TimeLogger & TimeLogger::getInstance()
{
    static TimeLogger instance;
    return instance;
}

TimeLogger::TimeLogger() : AbstractLogger("measuring time", std::ios::app)
{
    std::cout << "Logger "<< std::endl;
}

CompareLogger &CompareLogger::getInstance()
{
    static CompareLogger instance;
    return instance;
}

CompareLogger::CompareLogger() : AbstractLogger("comparer", std::ios::app)
{
}
