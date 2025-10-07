from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

db_path = 'postgresql://postgres:admin@localhost/all_pytorch_projects'

engine = create_engine(db_path)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()
