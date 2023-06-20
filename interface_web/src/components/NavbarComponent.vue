<template>
    <header class="bg-white dark:bg-gray-900">
      <div class="mx-auto max-w-screen-xl px-4 sm:px-6 lg:px-8">
        <div class="flex h-16 items-center justify-between">
          <div class="flex items-center">
            <a class="block text-teal-600 dark:text-teal-600" href="/">
              <img src="../assets/logo_copy.png" class="h-16 mr-2" alt="Logo">
            </a>
            <nav class="ml-4">
              <ul class="flex items-center gap-6 text-sm">
                <li>
                  <a
                    class="text-gray-500 transition hover:text-gray-500/75 dark:text-white dark:hover:text-white/75"
                    href="about.html"
                  >
                    About
                  </a>
                </li>
  
                <li>
                  <a
                    class="text-gray-500 transition hover:text-gray-500/75 dark:text-white dark:hover:text-white/75"
                    href="/"
                  >
                    Services
                  </a>
                </li>
  
                <li>
                  <a
                    class="text-gray-500 transition hover:text-gray-500/75 dark:text-white dark:hover:text-white/75"
                    href="/"
                  >
                    Projects
                  </a>
                </li>
  
                <li>
                  <a
                    class="text-gray-500 transition hover:text-gray-500/75 dark:text-white dark:hover:text-white/75"
                    href="/"
                  >
                    Blog
                  </a>
                </li>
              </ul>
            </nav>
          </div>
        </div>
      </div>
    </header>
  </template>
  
  <style>
  /* Header */
  header {
    background-color: #ffffff;
  }
  
  header a {
    color: #06b6d4;
  }
  
  header img {
    height: 70px;
  }
  
  nav ul {
    display: flex;
    justify-content: flex-end;
    gap: 1.5rem;
  }
  
  nav a {
    color: #6b7280;
    transition: color 0.3s;
  }
  
  nav a:hover {
    color: #4b5563;
  }
  
  /* Dark mode */
  .dark {
    background-color: #1f2937;
  }
  
  .dark a {
    color: #ffffff;
  }
  
  .dark a:hover {
    color: #d1d5db;
  }
  </style>
  