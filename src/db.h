#ifndef DB_H
#define DB_H

//#include <boost/tokenizer.hpp>
#include <iostream>
#include <map>
#include <sqlite3.h>
//#include <sstream>
#include <stdio.h>
//#include <vector>

#include "arcface-ir50.h"

#define FLOAT_BYTE 4

class Database {
  public:
    Database(std::string path, int embedding_dim);
    ~Database();
    //std::string insertPerson(std::string name);
    //std::string insertPersonIfNotExist(std::string name);
    int insertUser(std::string userId, std::string userName, std::string userFullName, std::string ofcCd, int active, int creDt, int creUser, int updDt, int updUser);
    //int insertFace(std::string userId, std::string image, float embedding[]);
    int insertFace(std::string userId, std::string imgName, float embedding[], int trainingFlag, int creDt, int creUser, int updDt, int updUser);
    int deleteUser(std::string userId);
    int deleteFace(int id);
    std::map<std::string, std::string> getUserDict();
    int getNumEmbeddings();
    int getEmbeddings(ArcFaceIR50 &recognizer);

  private:
    sqlite3 *m_db;
    int m_embedding_dim;
    //float *m_embedding;
};
#endif // DB_H
