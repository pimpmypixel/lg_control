import ntplib
import time

class NTPClient:
    def __init__(self):
        self.ntp_client = ntplib.NTPClient()
        self.time_offset = 0
        self.sync_ntp()
    
    def sync_ntp(self):
        """Synchronize with NTP server"""
        try:
            response = self.ntp_client.request('pool.ntp.org', version=3)
            self.time_offset = response.offset
            print(f"NTP sync successful. Offset: {self.time_offset:.3f}s")
        except Exception as e:
            print(f"NTP sync failed: {e}")
            self.time_offset = 0
    
    def get_ntp_time(self):
        """Get NTP synchronized timestamp"""
        return time.time() + self.time_offset 