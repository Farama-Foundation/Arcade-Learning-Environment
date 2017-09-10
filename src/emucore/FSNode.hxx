//============================================================================
//
//   SSSS    tt          lll  lll
//  SS  SS   tt           ll   ll
//  SS     tttttt  eeee   ll   ll   aaaa
//   SSSS    tt   ee  ee  ll   ll      aa
//      SS   tt   eeeeee  ll   ll   aaaaa  --  "An Atari 2600 VCS Emulator"
//  SS  SS   tt   ee      ll   ll  aa  aa
//   SSSS     ttt  eeeee llll llll  aaaaa
//
// Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
//
// See the file "license" for information on usage and redistribution of
// this file, and for a DISCLAIMER OF ALL WARRANTIES.
//
// $Id: FSNode.hxx,v 1.12 2007/08/07 14:38:51 stephena Exp $
//
//   Based on code from ScummVM - Scumm Interpreter
//   Copyright (C) 2002-2004 The ScummVM project
//============================================================================

#ifndef FS_NODE_HXX
#define FS_NODE_HXX

#include <algorithm>

/*
 * The API described in this header is meant to allow for file system browsing in a
 * portable fashions. To this ends, multiple or single roots have to be supported
 * (compare Unix with a single root, Windows with multiple roots C:, D:, ...).
 *
 * To this end, we abstract away from paths; implementations can be based on
 * paths (and it's left to them whether / or \ or : is the path separator :-);
 * but it is also possible to use inodes or vrefs (MacOS 9) or anything else.
 *
 * NOTE: Backends still have to provide a way to extract a path from a FSIntern
 *
 * You may ask now: "isn't this cheating? Why do we go through all this when we use
 * a path in the end anyway?!?".
 * Well, for once as long as we don't provide our own file open/read/write API, we
 * still have to use fopen(). Since all our targets already support fopen(), it should
 * be possible to get a fopen() compatible string for any file system node.
 *
 * Secondly, with this abstraction layer, we still avoid a lot of complications based on
 * differences in FS roots, different path separators, or even systems with no real
 * paths (MacOS 9 doesn't even have the notion of a "current directory").
 * And if we ever want to support devices with no FS in the classical sense (Palm...),
 * we can build upon this.
 */

/*
 * TODO - Instead of starting with getRoot(), we should rather add a getDefaultDir()
 * call that on Unix might return the current dir or the users home dir...
 * i.e. the root dir is usually not the best starting point for browsing.
 */

#include "../common/Array.hxx"

class FilesystemNode;

/**
 * List of multiple file system nodes. E.g. the contents of a given directory.
 */
class FSList : public Common::Array<FilesystemNode>
{
  public:
    void sort();
};


/**
 * File system node.
 */
class AbstractFilesystemNode
{
  public:
    /**
      Flag to tell listDir() which kind of files to list.
     */
    typedef enum {
      kListFilesOnly = 1,
      kListDirectoriesOnly = 2,
      kListAll = 3
    } ListMode;

    virtual ~AbstractFilesystemNode() {}

    /**
      Return display name, used by e.g. the GUI to present the file in the file browser.

      @return the display name
     */
    virtual std::string displayName() const = 0;

    /**
      Is this node valid (i.e. referring to an actual FS object)?
     */
    virtual bool isValid() const = 0;

    /**
      Is this node a directory or not?
     */
    virtual bool isDirectory() const = 0;

    /**
      A path representation suitable for use with fopen()
     */
    virtual std::string path() const = 0;

    /**
      List the content of this directory node.
      If this node is not a directory, throw an exception or call error().
     */
    virtual FSList listDir(ListMode mode = kListDirectoriesOnly) const = 0;

    /**
      Compare the name of this node to the name of another.
     */
    virtual bool operator< (const AbstractFilesystemNode& node) const
    {
      std::string first = displayName();
      std::string second = node.displayName();
      std::transform(first.begin(), first.end(), first.begin(), (int(*)(int)) tolower);
      std::transform(second.begin(), second.end(), second.begin(), (int(*)(int)) tolower);
      return first < second;
    }

    /**
      Test whether given path exists as a file.
     */
    static bool fileExists(const std::string& path);

    /**
      Test whether given path exists as a directory.
     */
    static bool dirExists(const std::string& path);

    /**
      Create a directory from the given path.
     */
    static bool makeDir(const std::string& path);

    /* TODO:
    bool isReadable();
    bool isWriteable();
    */

  protected:
    friend class FilesystemNode;

    /**
      The parent node of this directory.
      The parent of the root is the root itself.
     */
    virtual AbstractFilesystemNode *parent() const = 0;

    /**
     * This method is a rather ugly hack which is used internally by the
     * actual node implementions to wrap up raw nodes inside FilesystemNode
     * objects. We probably want to get rid of this eventually and replace it
     * with a cleaner / more elegant solution, but for now it works.
     * @note This takes over ownership of node. Do not delete it yourself,
     *       else you'll get ugly crashes. You've been warned!
     */
    static FilesystemNode wrap(AbstractFilesystemNode *node);
};


class FilesystemNode : public AbstractFilesystemNode
{
  friend class AbstractFilesystemNode;

  public:
    FilesystemNode();
    FilesystemNode(const FilesystemNode& node);
    FilesystemNode(const std::string& path);
    ~FilesystemNode();

    FilesystemNode &operator  =(const FilesystemNode &node);

    FilesystemNode getParent() const;
    bool hasParent() const;

    virtual std::string displayName() const { return _realNode->displayName(); }
    virtual bool isValid() const { return _realNode->isValid(); }
    virtual bool isDirectory() const { return _realNode->isDirectory(); }
    virtual std::string path() const { return _realNode->path(); }

    virtual FSList listDir(ListMode mode = kListDirectoriesOnly) const
      { return _realNode->listDir(mode); }

  protected:
    void decRefCount();

    virtual AbstractFilesystemNode* parent() const { return 0; }

  private:
    AbstractFilesystemNode *_realNode;
    int *_refCount;

    /**
     * Returns a special node representing the FS root. The starting point for
     * any file system browsing.
     * On Unix, this will be simply the node for / (the root directory).
     * On Windows, it will be a special node which "contains" all drives (C:, D:, E:).
     */
    static AbstractFilesystemNode* getRoot();

    /*
     * Construct a node based on a path; the path is in the same format as it
     * would be for calls to fopen().
     *
     * I.e. getNodeForPath(oldNode.path()) should create a new node identical to oldNode.
     */
    static AbstractFilesystemNode* getNodeForPath(const std::string& path);
};

#endif
