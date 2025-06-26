    git clone https://github.com/haroldgomez/SupportModel.git
    cd SupportModel
    ```

2.  **Install dependencies**:
    (Assuming Python and pip are installed)
    ```bash
    pip install -r requirements.txt
    ```

3.  **Database Setup**:
    (Instructions will vary based on the database used, e.g., PostgreSQL, MySQL, SQLite)
    -   Configure your database connection in `config.py` (or equivalent).
    -   Run database migrations:
        ```bash
        python manage.py migrate
        ```

4.  **Run the application**:
    ```bash
    streamlit run streamlit_qa_app.py
   
    