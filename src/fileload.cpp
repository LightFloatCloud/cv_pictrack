
#include "fileload.h"

#include <algorithm>
#include <dirent.h>


bool cmp(std::string const &arg_a, std::string const &arg_b) {
    return arg_a.size() < arg_b.size() || (arg_a.size() == arg_b.size() && arg_a < arg_b);
}

bool LoadFileList(const std::string path, const std::string suffix, std::vector<std::string>& file_list) 
{
    DIR* dir;
    struct dirent* ent;
    std::vector<std::string> names;
    if ((dir = opendir(path.c_str())) != NULL)
    {
        std::cout<< "Success Open dir: " << path << std::endl;
        while ((ent = readdir(dir)) != NULL)
        {
            /* print all the files and directories within directory */
            // printf("%s\n", ent->d_name);
            std::string filename = ent->d_name;
            std::cout << "Find file: " << filename << std::endl;
            std::string filetype = filename.substr(filename.rfind("."), filename.length());
            if(filetype == suffix )
            {
                names.push_back(filename);
                //cout << "Load file: " << filename << endl;
            }
            else
            {
                std::cout << "Ignore." << std::endl;
            }
        }
        closedir(dir);
    }
    else
    {
        std::cout<< "Cannot Open dir: " << path <<std::endl;
        return false;
    }
    /* sort by name */
    std::sort(names.begin(), names.end(), cmp);

    std::cout << std::endl;
    file_list = names;
    for(auto it=file_list.begin();it!=file_list.end();it++)
    {
        std::cout <<"Load file: "<< *it <<std::endl;
    }
    return true;
}







