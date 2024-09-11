USE ROLE ACCOUNTADMIN;

-- Using ACCOUNTADMIN, create a new role for this exercise and grant to applicable users
-- Replace <YOUR_USER> with your Snowflake username
CREATE OR REPLACE ROLE FS_DBT_ROLE;
GRANT ROLE FS_DBT_ROLE to USER <YOUR_USER>;

-- create our virtual warehouse
CREATE OR REPLACE WAREHOUSE FS_DBT_WH AUTO_SUSPEND = 60;

GRANT ALL ON WAREHOUSE FS_DBT_WH TO ROLE FS_DBT_ROLE;

-- Next create a new database and schema,
CREATE OR REPLACE DATABASE FS_DBT_DATABASE;
CREATE OR REPLACE SCHEMA FS_DBT_SCHEMA;

GRANT OWNERSHIP ON DATABASE FS_DBT_DATABASE TO ROLE FS_DBT_ROLE COPY CURRENT GRANTS;
GRANT OWNERSHIP ON ALL SCHEMAS IN DATABASE FS_DBT_DATABASE TO ROLE FS_DBT_ROLE COPY CURRENT GRANTS;