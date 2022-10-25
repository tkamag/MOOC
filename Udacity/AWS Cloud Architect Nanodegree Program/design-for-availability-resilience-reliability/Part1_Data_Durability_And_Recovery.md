## Data Durability And Recovery

1. Pick two AWS regions. An active region and a standby region (only **us-east-1** and **us-west-2** regions are allowed)

2. Use CloudFormation to create one VPC in each region. Name the VPC in the active region **Primary** and name the VPC in the standby region **Secondary**.

**NOTE**: Be sure to use different CIDR address ranges for the VPCs.

**SAVE**: Screenshots of both VPCs after they are created. Name your screenshots:

<figure>
  <img src="./fig/00-primary_VPC.png" alt=".." title="Optional title" width="60%" height="70%"/>  
 <figcaption></figcaption>
</figure>

<figure>
  <img src="./fig/01-secondary_VPC.png" alt=".." title="Optional title" width="60%" height="70%"/>  
 <figcaption></figcaption>
</figure>