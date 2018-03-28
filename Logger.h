#pragma once

#include <iostream>
#include <fstream>

class AbstractLogger
{
public:
    template< typename T>
    AbstractLogger & operator << (const T & itemToLog);
    template< typename T>
    AbstractLogger & operator >> (const T & itemToLog);


protected:
    std::fstream logFile;

             AbstractLogger(std::string name, std::ios_base::openmode  mode);
    virtual ~AbstractLogger();
};

template<typename T>
inline AbstractLogger & AbstractLogger::operator<<(const T & itemToLog)
{
    logFile << itemToLog;
    return *this;
}

template<typename T>
inline AbstractLogger & AbstractLogger::operator>>(const T & itemToConsole)
{
    logFile << itemToConsole;
    return *this;
}

class ErrorLogger : public AbstractLogger
{
public:
    static ErrorLogger& getInstance();

    ErrorLogger();
};

class TimeLogger : public AbstractLogger
{
public:
    static TimeLogger& getInstance();

    TimeLogger();
};

class CompareLogger : public AbstractLogger
{
public:
    static CompareLogger& getInstance();

    CompareLogger();
};
