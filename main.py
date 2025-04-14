from fastapi import FastAPI, HTTPException
from sqlalchemy import inspect, text
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = FastAPI(title="Database API")

# Get database credentials from environment variables
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")  # Password will be loaded from .env file
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# Construct the database URL
def get_db_url():
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"

@app.get("/")
def read_root():
    return {"message": "Welcome to the API"}

@app.get("/pg-cm-qa-db1-tables")
def get_pg_cm_qa_db1_tables():
    try:
        # Create a new connection to the pg-cm-qa-db1 database
        from sqlalchemy import create_engine
        
        # Use the function to get the database URL
        db_url = get_db_url()
        
        # Create a new engine for this specific database
        pg_cm_qa_db1_engine = create_engine(db_url)
        
        # Create a connection
        with pg_cm_qa_db1_engine.connect() as conn:
            # Get all tables in the database
            inspector = inspect(pg_cm_qa_db1_engine)
            table_names = inspector.get_table_names()
            
            # Get schema information
            result = conn.execute(text("SELECT schema_name FROM information_schema.schemata"))
            schemas = [row[0] for row in result]
            
            # Get detailed table information including schemas
            tables_info = []
            for schema in schemas:
                schema_tables = inspector.get_table_names(schema=schema)
                for table in schema_tables:
                    tables_info.append({"schema": schema, "table": table})
            
            logger.info(f"Tables in pg-cm-qa-db1: {table_names}")
            
            return {
                "database": "pg-cm-qa-db1",
                "schemas": schemas,
                "tables": table_names,
                "detailed_tables": tables_info
            }
    except Exception as e:
        error_msg = f"Failed to get tables from pg-cm-qa-db1: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)