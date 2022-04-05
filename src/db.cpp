#include "db.h"

Database::Database(std::string path, int embedding_dim) {
    m_embedding_dim = embedding_dim;

    std::string sql;
    sqlite3_stmt *stmt;
    int rc;
    int isNotEmpty = 0;

    std::cout << "Database path: " << path << "\n";
    rc = sqlite3_open(path.c_str(), &m_db);
    if (rc) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(m_db));
    } else {
        fprintf(stderr, "Opened database successfully\n");
    }

    // Prepare
    sql = "SELECT name FROM sqlite_master;";
    rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
    }

    // Step
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        isNotEmpty = 1;
        break;
    }
    sqlite3_finalize(stmt);
    //std::cout << rc <<  " " << isNotEmpty << " 111\n";

    if (isNotEmpty) {
        std::cout << "Database not empty\n";
    } else {
        std::cout << "Empty Database, initiating...\n";

        // Prepare
        sql = "CREATE TABLE IF NOT EXISTS HISTORY ( \
                HIS_ID INTEGER PRIMARY KEY AUTOINCREMENT, \
                USR_ID INTEGER, \
                PUNCH_TIME INTEGER, \
                CRE_DT INTEGER, \
                CRE_USR INTEGER, \
                UPD_DT INTEGER, \
                UPD_USR INTEGER \
                )";
        rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
        if (rc != SQLITE_OK) {
            std::cout << "SQL error: " << sqlite3_errmsg(m_db);
        }

        // Step
        rc = sqlite3_step(stmt);
        if (rc != SQLITE_ROW && rc != SQLITE_DONE) {
            std::cout << "SQL step error: " << sqlite3_errmsg(m_db);
        } else {
            std::cout << "Table `HISTORY` created successfully\n";
        }
        sqlite3_finalize(stmt);

        // Prepare
        sql = "CREATE TABLE IF NOT EXISTS USERS ( \
                USR_ID TEXT PRIMARY KEY, \
                USR_NM TEXT, \
                FULL_NM TEXT, \
                OFC_CD TEXT, \
                ACTIVE INTEGER, \
                CRE_DT INTEGER, \
                CRE_USR INTEGER, \
                UPD_DT INTEGER, \
                UPD_USR INTEGER \
                )";
        rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
        if (rc != SQLITE_OK) {
            std::cout << "SQL error: " << sqlite3_errmsg(m_db);
        }

        // Step
        rc = sqlite3_step(stmt);
        if (rc != SQLITE_ROW && rc != SQLITE_DONE) {
            std::cout << "SQL step error: " << sqlite3_errmsg(m_db);
        } else {
            std::cout << "Table `USERS` created successfully\n";
        }
        sqlite3_finalize(stmt);

        // Prepare
        sql = "CREATE TABLE IF NOT EXISTS IMAGES_USER ( \
                IMG_ID INTEGER PRIMARY KEY AUTOINCREMENT, \
                USR_ID TEXT, \
                IMG_NM TEXT, \
                EMBEDDING BLOB, \
                TRAINING_FLG INTEGER, \
                CRE_DT INTEGER, \
                CRE_USR INTEGER, \
                UPD_DT INTEGER, \
                UPD_USR INTEGER, \
                UNIQUE(IMG_ID, USR_ID), \
                FOREIGN KEY(USR_ID) REFERENCES USERS(USR_ID) \
                )";

        rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
        if (rc != SQLITE_OK) {
            std::cout << "SQL error: " << sqlite3_errmsg(m_db);
        }

        // Step
        rc = sqlite3_step(stmt);
        if (rc != SQLITE_ROW && rc != SQLITE_DONE) {
            std::cout << "SQL step error: " << sqlite3_errmsg(m_db);
        } else {
            std::cout << "Table `IMAGES_USER` created successfully\n";
        }
        sqlite3_finalize(stmt);
    }
}

int Database::insertUser(std::string userId, std::string userName, std::string userFullName, std::string ofcCd,
                         int active, int creDt, int creUser, int updDt, int updUser) {
    std::string sql;
    sqlite3_stmt *stmt;
    int rc;

    // Prepare
    sql = "INSERT INTO USERS (USR_ID, USR_NM, FULL_NM, OFC_CD, ACTIVE, CRE_DT, CRE_USR, UPD_DT, UPD_USR) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);";
    rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
        return -1;
    }

    // Binding
    rc = sqlite3_bind_text(stmt, 1, userId.c_str(), userId.length(), SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }
    rc = sqlite3_bind_text(stmt, 2, userName.c_str(), userName.length(), SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }
    rc = sqlite3_bind_text(stmt, 3, userFullName.c_str(), userFullName.length(), SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }
    rc = sqlite3_bind_text(stmt, 4, ofcCd.c_str(), ofcCd.length(), SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }
    rc = sqlite3_bind_int(stmt, 5, active);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }
    rc = sqlite3_bind_int(stmt, 6, creDt);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }
    rc = sqlite3_bind_int(stmt, 7, creUser);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }
    rc = sqlite3_bind_int(stmt, 8, updDt);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }
    rc = sqlite3_bind_int(stmt, 9, updUser);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }

    // Step
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW && rc != SQLITE_DONE) {
        std::cout << "SQL step error: " << sqlite3_errmsg(m_db);
        sqlite3_finalize(stmt);
        return -1;
    }
    if (rc == SQLITE_DONE) {
        std::cout << "Record for user `" << userId << "` created successfully\n";
    }
    sqlite3_finalize(stmt);
    return 1;
}

int Database::insertFace(std::string userId, std::string imgName, float embedding[], int trainingFlag, int creDt,
                         int creUser, int updDt, int updUser) {
    std::string sql;
    sqlite3_stmt *stmt;
    int rc;

    // Prepare
    sql = "INSERT INTO IMAGES_USER (USR_ID, IMG_NM, EMBEDDING, TRAINING_FLG, CRE_DT, CRE_USR, UPD_DT, UPD_USR) VALUES (?, ?, ?, ?, ?, ?, ?, ?);";
    rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
        return -1;
    }

    // Binding
    rc = sqlite3_bind_text(stmt, 1, userId.c_str(), userId.length(), SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }
    rc = sqlite3_bind_text(stmt, 2, imgName.c_str(), imgName.length(), SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }
    rc = sqlite3_bind_blob(stmt, 3, embedding, FLOAT_BYTE * m_embedding_dim, SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }
    rc = sqlite3_bind_int(stmt, 4, trainingFlag);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }
    rc = sqlite3_bind_int(stmt, 5, creDt);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }
    rc = sqlite3_bind_int(stmt, 6, creUser);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }
    rc = sqlite3_bind_int(stmt, 7, updDt);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }
    rc = sqlite3_bind_int(stmt, 8, updUser);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -2;
    }

    // Step
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW && rc != SQLITE_DONE) {
        std::cout << "SQL step error: " << sqlite3_errmsg(m_db);
        sqlite3_finalize(stmt);
        return -1;
    }
    if (rc == SQLITE_DONE) {
        std::cout << "Embedding for user id `" << userId << "` created successfully\n";
    }
    sqlite3_finalize(stmt);
    return 1;
}

int Database::deleteFace(int id) {
    std::string sql;
    sqlite3_stmt *stmt;
    int rc;

    // Prepare
    sql = "DELETE FROM IMAGES_USER WHERE IMG_ID=?;";
    rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
        return -1;
    }

    // Binding
    rc = sqlite3_bind_int(stmt, 1, id);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }

    // Step
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW && rc != SQLITE_DONE) {
        std::cout << "SQL step error: " << sqlite3_errmsg(m_db);
        sqlite3_finalize(stmt);
        return -1;
    }
    if (rc == SQLITE_DONE) {
        std::cout << "Embedding with id `" << id << "` deleted successfully\n";
    }
    sqlite3_finalize(stmt);
    return 0;
}

int Database::deleteUser(std::string userId) {
    std::string sql;
    sqlite3_stmt *stmt;
    int rc;

    // Delete embeddings
    // Prepare
    sql = "DELETE FROM IMAGES_USER WHERE USR_ID=?;";
    rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
        return -1;
    }

    // Binding
    rc = sqlite3_bind_text(stmt, 1, userId.c_str(), userId.length(), SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }

    // Step
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW && rc != SQLITE_DONE) {
        std::cout << "SQL step error: " << sqlite3_errmsg(m_db);
        sqlite3_finalize(stmt);
        return -1;
    }
    if (rc == SQLITE_DONE) {
        std::cout << "Deleted all embeddings with user id `" << userId << "` successfully\n";
    }
    sqlite3_finalize(stmt);

    // Delete person
    // Prepare
    sql = "DELETE FROM USERS WHERE USR_ID=?;";
    rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
        return -1;
    }

    // Binding
    rc = sqlite3_bind_text(stmt, 1, userId.c_str(), userId.length(), SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }

    // Step
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW && rc != SQLITE_DONE) {
        std::cout << "SQL step error: " << sqlite3_errmsg(m_db);
        sqlite3_finalize(stmt);
        return -1;
    }
    if (rc == SQLITE_DONE) {
        std::cout << "Deleted person with id `" << userId << "` successfully\n";
    }
    sqlite3_finalize(stmt);
    return 0;
}

std::map<std::string, std::string> Database::getUserDict() {
    std::map<std::string, std::string> userDict;
    std::string sql;
    sqlite3_stmt *stmt;
    int rc;
    std::string userId;
    std::string userName;

    // get user id and name
    sql = "SELECT * FROM USERS;";

    rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
    }
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        //id = sqlite3_column_int(stmt, 0);
        userId = std::string(reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0)));
        userName = std::string(reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1)));
        userDict[userId] = userName;
    }
    //std::cout << userDict.size() << " 2222222222222222222\n";
    if (rc != SQLITE_DONE) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
    }
    sqlite3_finalize(stmt);
    return userDict;
}

int Database::getNumEmbeddings() {
    int numEmbeds = 0;
    std::string sql;
    sqlite3_stmt *stmt;
    int rc;

    // Prepare
    sql = "SELECT COUNT(*) FROM IMAGES_USER;";
    rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
        return -1;
    }
    // Step
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW && rc != SQLITE_DONE) {
        std::cout << "SQL step error: " << sqlite3_errmsg(m_db);
        sqlite3_finalize(stmt);
        return -1;
    }
    numEmbeds = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    return numEmbeds;
}

int Database::getEmbeddings(ArcFaceIR50 &recognizer) {
    std::string sql;
    sqlite3_stmt *stmt;
    int rc;
    std::string userId;
    float *embedding;

    // dict that map user id to user name
    //std::map<std::string, std::string> userDict = getUserDict();

    // init num embeds
    int numEmbeds = getNumEmbeddings();
    std::cout << "There are " << numEmbeds << " embeddings in database\n";
    recognizer.initKnownEmbeds(numEmbeds);

    // get embedding
    sql = "SELECT * FROM IMAGES_USER;";

    rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
        return -1;
    }
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        userId = std::string(reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1)));
        embedding = (float *)sqlite3_column_blob(stmt, 3);
        //std::cout << userId << " " << embedding[0] << "\n";
        recognizer.addEmbedding(userId, embedding);
    }
    if (rc != SQLITE_DONE) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
    }
    sqlite3_finalize(stmt);
    return 0;
}

/*
static int callback(void *NotUsed, int argc, char **argv, char **azColName) {
    int i;
    for (i = 0; i < argc; i++) {
        printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
    }
    printf("\n");
    return 0;
}

int Database::insertFaceString(int personId, std::string image, float embedding[]) {
    std::string sql;
    char *zErrMsg = 0;
    int rc;

    std::string embedding_str = embed2str(embedding);

    // Create SQL statement
    sql = "INSERT INTO face (person_id, image, embedding) VALUES (" + std::to_string(personId) + ", \"" + image + "\"" +
          ", \"" + embedding_str + "\");";
    std::cout << "insert sql: " << sql << "\n";

    // Execute SQL statement
    rc = sqlite3_exec(m_db, sql.c_str(), callback, 0, &zErrMsg);

    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", zErrMsg);
        sqlite3_free(zErrMsg);
    } else {
        fprintf(stdout, "Record created successfully\n");
    }

    return rc;
}

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

std::string Database::embed2str(float embedding[]){
    //std::string embedding_str = std::to_string(embedding[0]);
    std::string embedding_str = to_string_with_precision(embedding[0], 24);
    for (int i = 1; i < m_embedding_dim; i++) {
        //embedding_str += "," + std::to_string(embedding[i]);
        embedding_str += "," + to_string_with_precision(embedding[i], 24);
    }
    return embedding_str;
}

float *Database::str2embed(std::string str){
    boost::char_separator<char> sep(",");
    boost::tokenizer<boost::char_separator<char>> tokens(str, sep);

    int i = 0;
    for (const auto& t : tokens) {
        m_embedding[i] = std::strtof(t.c_str(), NULL);
        i += 1;
    }
    return m_embedding;
}

std::string Database::insertPerson(std::string name) {
    std::string sql;
    sqlite3_stmt *stmt;
    int rc;

    // Prepare
    sql = "INSERT INTO USERS (USR_ID) VALUES (?);";
    rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
        return "";
    }

    // Binding
    rc = sqlite3_bind_text(stmt, 1, name.c_str(), name.length(), SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return "";
    }

    // Step
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW && rc != SQLITE_DONE) {
        std::cout << "SQL step error: " << sqlite3_errmsg(m_db);
        sqlite3_finalize(stmt);
        return "";
    }
    if (rc == SQLITE_DONE) {
        std::cout << "Record for person `" << name << "` created successfully\n";
    }
    sqlite3_finalize(stmt);

    // get last ID
    //int last_id = sqlite3_last_insert_rowid(m_db);
    return name;
}

std::string Database::insertPersonIfNotExist(std::string name) {
    std::string sql;
    sqlite3_stmt *stmt;
    int rc;
    //int id = 0;
    std::string id = "";

    //sql = "SELECT id FROM person WHERE name=\"" + name + "\";";
    // Prepare
    sql = "SELECT USR_ID FROM USERS WHERE USR_ID=?;";

    rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
        return "";
    }
    // Bind
    rc = sqlite3_bind_text(stmt, 1, name.c_str(), name.length(), SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return "";
    }

    // Step
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        //id = sqlite3_column_int(stmt, 0);
        id = std::string(reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0)));
    }
    if (rc != SQLITE_DONE) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
    }
    sqlite3_finalize(stmt);

    if (id != "") {
        std::cout << name + " already in database\n";
    } else {
        id = insertPerson(name);
    }
    return id;
}

int Database::insertFace(std::string userId, std::string image, float embedding[]) {
    std::string sql;
    sqlite3_stmt *stmt;
    int rc;

    // Prepare
    sql = "INSERT INTO IMAGES_USER (USR_ID, IMG_NM, EMBEDDING) VALUES (?, ?, ?);";
    rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
        return -1;
    }

    // Binding
    rc = sqlite3_bind_text(stmt, 1, userId.c_str(), userId.length(), SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }
    rc = sqlite3_bind_text(stmt, 2, image.c_str(), image.length(), SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }
    rc = sqlite3_bind_blob(stmt, 3, embedding, FLOAT_BYTE * m_embedding_dim, SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -1;
    }

    // Step
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW && rc != SQLITE_DONE) {
        std::cout << "SQL step error: " << sqlite3_errmsg(m_db);
        sqlite3_finalize(stmt);
        return -1;
    }
    if (rc == SQLITE_DONE) {
        std::cout << "Embedding for user id `" << userId << "` created successfully\n";
    }
    sqlite3_finalize(stmt);
    return 0;
}


*/

Database::~Database() { sqlite3_close(m_db); }
