/* ######################################################### 
CREATE DATABASE, SCHEMA, AND STAGES
######################################################### */
use role sysadmin;


create or alter database sf_clinical_db;
create or alter schema sf_clinical_db.results;
create or alter schema sf_clinical_db.utils;

create or replace stage monai_medical_images_stg 
encryption = (type = 'SNOWFLAKE_SSE')
directory = (enable=true);

create or replace stage results_stg 
encryption = (type = 'SNOWFLAKE_SSE')
directory = (enable=true);

/* ######################################################### 
ASSIGN INTEGRATION AND COMPUTE POOL PRIVILEGES TO SYSADMIN ROLE
######################################################### */
use role accountadmin;
grant create integration on account to role sysadmin;
grant create compute pool on account to role sysadmin;


/* ######################################################### 
CREATE NETWORK RULE AND EXTERNAL ACCESS INTEGRATION
######################################################### */
use role sysadmin;

use schema sf_clinical_db.utils;

create network rule if not exists allow_all_network_rules
  mode = egress 
  type = host_port
  value_list = ('0.0.0.0');
;

CREATE EXTERNAL ACCESS INTEGRATION IF NOT EXISTS ALLOW_ALL_EAI 
  ALLOWED_NETWORK_RULES = (allow_all_network_rules)
  ENABLED = true
;

CREATE COMPUTE POOL IF NOT EXISTS GPU_ML_M_POOL 
  min_nodes = 1
  max_nodes = 8 
  instance_family = 'GPU_NV_M'
;
