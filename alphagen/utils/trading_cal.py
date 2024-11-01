from datetime import datetime
import exchange_calendars as xcals
import pandas as pd

cal = xcals.get_calendar("XNYS")  # New York Stock Exchange
print( cal.sessions_in_range("20100101", "20241028") )
#for date in cal.sessions_in_range("2010-01-01", "2024-10-28"):
#    print( date.strftime("%Y%m%d") )

print( cal.sessions_window("2022-01-03", -10) )
print( cal.sessions_window( cal.previous_session("2022-01-03"), -10) )
print( cal.previous_session('20220104') )
print( cal.next_session('20220104') )
session = '2022-01-04'
print( cal.session_offset(session, count=-1), cal.session_offset(session, count=1) )
