#include "db.h"

Database::Database(std::string path, int embedding_dim) {
    m_embedding_dim = embedding_dim;

    std::string sql;
    sqlite3_stmt *stmt;
    int rc;
    int isNotEmpty = 0;

    std::cout << "[INFO] Database path: " << path << "\n";
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

    if (isNotEmpty) {
        std::cout << "Database not empty\n";
    } else {
        std::cout << "Empty Database, initiating...\n";

        // Prepare
        sql = "CREATE TABLE IF NOT EXISTS USER ( \
                USR_ID TEXT PRIMARY KEY, \
                USR_NM TEXT \
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
            std::cout << "Table `USER` created successfully\n";
        }
        sqlite3_finalize(stmt);

        // Prepare
        sql = "CREATE TABLE IF NOT EXISTS FACE ( \
                IMG_ID    INTEGER PRIMARY KEY AUTOINCREMENT, \
                USR_ID    TEXT, \
                IMG_PATH  TEXT, \
                EMBEDDING BLOB, \
                UNIQUE(IMG_ID, USR_ID), \
                FOREIGN KEY(USR_ID) REFERENCES USER(USR_ID) \
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
            std::cout << "Table `FACE` created successfully\n";
        }
        sqlite3_finalize(stmt);
    }
}

int Database::insertUser(std::string userId, std::string userName) {
    std::string sql;
    sqlite3_stmt *stmt;
    int rc;

    // Prepare
    sql = "INSERT INTO USER (USR_ID, USR_NM) VALUES (?, ?);";
    rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
        return -1;
    }

    // Binding
    rc = sqlite3_bind_text(stmt, 1, userId.c_str(), userId.length(), SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -2;
    }
    rc = sqlite3_bind_text(stmt, 2, userName.c_str(), userName.length(), SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -2;
    }

    // Step
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW && rc != SQLITE_DONE) {
        std::cout << "SQL step error: " << sqlite3_errmsg(m_db) << ", error code: " << rc << "\n";
        sqlite3_finalize(stmt);
        return -3;
    }
    if (rc == SQLITE_DONE) {
        std::cout << "Record for user `" << userId << "` created successfully\n";
    }
    sqlite3_finalize(stmt);
    return 1;
}

int Database::insertFace(std::string userId, std::string imgPath, float embedding[]) {
    std::string sql;
    sqlite3_stmt *stmt;
    int rc;

    // Prepare
    sql = "INSERT INTO FACE (USR_ID, IMG_PATH, EMBEDDING) VALUES (?, ?, ?);";
    rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
        return -1;
    }

    // Binding
    rc = sqlite3_bind_text(stmt, 1, userId.c_str(), userId.length(), SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -2;
    }
    rc = sqlite3_bind_text(stmt, 2, imgPath.c_str(), imgPath.length(), SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -2;
    }
    rc = sqlite3_bind_blob(stmt, 3, embedding, sizeof(float) * m_embedding_dim, SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -2;
    }

    // Step
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW && rc != SQLITE_DONE) {
        std::cout << "SQL step error: " << sqlite3_errmsg(m_db) << ", error code: " << rc << "\n";
        sqlite3_finalize(stmt);
        return -3;
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
        return -2;
    }

    // Step
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW && rc != SQLITE_DONE) {
        std::cout << "SQL step error: " << sqlite3_errmsg(m_db);
        sqlite3_finalize(stmt);
        return -3;
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
    sql = "DELETE FROM FACE WHERE USR_ID=?;";
    rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
        return -1;
    }

    // Binding
    rc = sqlite3_bind_text(stmt, 1, userId.c_str(), userId.length(), SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -2;
    }

    // Step
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW && rc != SQLITE_DONE) {
        std::cout << "SQL step error: " << sqlite3_errmsg(m_db);
        sqlite3_finalize(stmt);
        return -3;
    }
    if (rc == SQLITE_DONE) {
        std::cout << "Deleted all embeddings with user id `" << userId << "` successfully\n";
    }
    sqlite3_finalize(stmt);

    // Delete person
    // Prepare
    sql = "DELETE FROM USER WHERE USR_ID=?;";
    rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
        return -1;
    }

    // Binding
    rc = sqlite3_bind_text(stmt, 1, userId.c_str(), userId.length(), SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        std::cout << "SQL bind error: " << sqlite3_errmsg(m_db);
        return -2;
    }

    // Step
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW && rc != SQLITE_DONE) {
        std::cout << "SQL step error: " << sqlite3_errmsg(m_db);
        sqlite3_finalize(stmt);
        return -3;
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
    sql = "SELECT * FROM USER;";

    rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
    }
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        // id = sqlite3_column_int(stmt, 0);
        userId = std::string(reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0)));
        userName = std::string(reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1)));
        userDict[userId] = userName;
    }
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
    sql = "SELECT COUNT(*) FROM FACE;";
    rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
        return -1;
    }
    // Step
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW && rc != SQLITE_DONE) {
        std::cout << "SQL step error: " << sqlite3_errmsg(m_db) << ", error code: " << rc << "\n";
        sqlite3_finalize(stmt);
        return -2;
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

    // init num embeds
    int numEmbeds = getNumEmbeddings();
    std::cout << "[INFO] There are " << numEmbeds << " embeddings in database\n";
    recognizer.initKnownEmbeds(numEmbeds);

    // get embedding
    sql = "SELECT * FROM FACE;";

    rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
        return -1;
    }
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        userId = std::string(reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1)));
        embedding = (float *)sqlite3_column_blob(stmt, 3);
        recognizer.addEmbedding(userId, embedding);
    }
    if (rc != SQLITE_DONE) {
        std::cout << "SQL error: " << sqlite3_errmsg(m_db);
    }
    sqlite3_finalize(stmt);
    return 0;
}

Database::~Database() { sqlite3_close(m_db); }
