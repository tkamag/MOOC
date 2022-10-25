## Minimum RTO for a single AZ outage

| Time  | Description                                                |
| ----- | ---------------------------------------------------------- |
| 00:02 | Problem occurred (First minute)                             |
| 00:12 | Time take to raise alerts (10 mins) |
| 00:13 | Automatically switch to second availability zone (1 mins)  |

**Total time 13 mins**

## Minimum RTO for a single region outage

| Time  | Description                                                                                 |
| ----- | ------------------------------------------------------------------------------------------- |
| 00:00 | Problem occurred (First minute)                                                              |
| 00:10 | Time take to raise alerts (10 mins)                                  |
| 00:11 | Alert triggers on-all staff (1 mins)                                                        |
| 00:21 | On-call staff may need to get out of bed, get to a computer, log in, log onto VPN (10 mins) |
| 00:31 | On-call Root cause Analysis starts (10 mins)                                         |
| 00:41 | Root cause Analysis done (10 mins)                                                          |
| 00:51 | Remediation started (10 mins)                                                               |
| 00:56 | Issue fixed (5 mins)                                                                        |

**Total time 57 mins**

## Minimum RPO for a single AZ outage

**RDS** has **point in time restore** available after every 5 min. At the most data loss will be of 5 minutes. 

## Minimum RPO for a single region outage

Read replicas are almost kept in sync with the primary database, ideally, they will be having at least the same **RPO** as the primary database. For the practical purpose we can consider at most the **RPO** should be around 13 mins.