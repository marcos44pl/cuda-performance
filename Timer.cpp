#include "Timer.h"
#include "Logger.h"
#include <iostream>
#include <numeric>
#include <algorithm>

Timer &Timer::getInstance()
{
    static Timer instance;
    return instance;
}

void Timer::start(std::string const& name)
{
    ResultMapIt it = resultMap.find(name);

    if (it == resultMap.end())
        resultMap[name] = ResultsData(timeNow());
    else
    {
        resultMap[name].startPoint = timeNow();
    }
}

long long Timer::stop(std::string const& name)
{
    ResultMapIt it = resultMap.find(name);
    long long dur = 0;
    if (it != resultMap.end())
    {
        dur = duration(timeNow() - resultMap[name].startPoint);
        resultMap[name].results.push_back(dur);
    }
    return dur;
}

long long Timer::stop()
{
    return duration(timeNow() - startTime);
}


void Timer::printResults()
{
    for (ResultMapIt it = resultMap.begin(); it != resultMap.end(); ++it)
    {
    	auto const& res = it->second.results;
        auto sum = std::accumulate(res.begin(),res.end(),0);
        auto max = std::max_element(res.begin(),res.end());
        auto min = std::min_element(res.begin(),res.end());
        long long avarage = sum / res.size();
        std::cout << it->first.c_str() << ":\n" <<
            "    Avarage execution time: " << avarage << "ms\n" <<
            "    Min execution time: " << *min << "ms\n" <<
            "    Max execution time: " << *max << "ms\n" <<
            "    Total execution time: " << sum << "ms\n" <<
            "    Executions count: " << res.size() << "\n";
    }
}

void Timer::updateMap(std::string const name, long long value)
{
    ResultMapIt it = resultMap.find(name);

    if (it == resultMap.end())
        resultMap[name] = ResultsData(timeNow());
    else
    {
        resultMap[name].results.push_back(value);
    }
}

long long Timer::getAvgResult(std::string const& name)
{
    ResultMapIt it = resultMap.find(name);
    if (it == resultMap.end())
           return 0;
    else
    {
    	auto const& res = it->second.results;
        auto sum = std::accumulate(res.begin(),res.end(),0);
        return sum / res.size();
    }
}
